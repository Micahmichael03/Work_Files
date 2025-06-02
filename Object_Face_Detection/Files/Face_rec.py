# Face Recognition System with QR Code Email Delivery

# Import required libraries 
import cv2  # OpenCV for image processing and camera operations
import face_recognition  # Library for face detection and recognition
import pyodbc  # For SQL Server database connection
import numpy as np  # For numerical operations and array handling
import base64  # For encoding/decoding images to base64
import os  # For file and directory operations
import webbrowser  # For opening web URLs
import time  # For time-related operations
import re  # For regular expressions
from datetime import datetime  # For timestamp operations
import qrcode  # For generating QR codes
from pyzbar.pyzbar import decode  # For decoding QR codes
from PIL import Image  # For image processing
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Database connection string - connects to Microsoft Fabric SQL Server
connection_string = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=gjmfi7jmo2delewe55pp7ledge-q64xenopcmje5bzhzccghkr4bu.database.fabric.microsoft.com,1433;"
    "Database=warehouseDB-5606c843-5230-4432-9741-392553ea9fd5;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Authentication=ActiveDirectoryPassword;"
    "UID=Micahmichael@makoflash02gmail.onmicrosoft.com;"
    "PWD=@Chukwuemeka2025"
)

# Try to establish database connection
try:
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
except Exception as e:
    print(f"Database connection error: {e}")
    exit()

# Add qr_code_base64 column to users table if it doesn't exist
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                  WHERE TABLE_NAME = 'users' AND COLUMN_NAME = 'qr_code_base64')
    ALTER TABLE users ADD qr_code_base64 TEXT
""")
conn.commit()

# Add email column to users table if not exists
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                  WHERE TABLE_NAME = 'users' AND COLUMN_NAME = 'email')
    ALTER TABLE users ADD email VARCHAR(255)
""")
conn.commit()

# Global lists to store known face data in memory
known_faces = []  # Stores face encodings
known_ids = []    # Stores corresponding user IDs
known_usernames = []  # Stores corresponding usernames

def format_timestamp():
    """Returns current timestamp in MM/DD/YYYY HH:MM AM/PM format"""
    return datetime.now().strftime("%m/%d/%Y %I:%M %p")

def generate_user_id(username):
    """
    Generates a unique user ID based on username
    Format: username_number (e.g., john_1, john_2)
    """
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return None, "Username can only contain letters, numbers, or underscores"
    try:
        cursor.execute("""
            SELECT MAX(CAST(SUBSTRING(username, LEN(?) + 2, LEN(username)) AS INT))
            FROM users 
            WHERE username LIKE ?
        """, (username, f"{username}_%"))
        result = cursor.fetchone()[0]
        next_number = 1 if result is None else result + 1
        return f"{username}_{next_number}", None
    except Exception as e:
        return None, f"Error generating user_id: {e}"

def load_known_faces():
    """Loads all known faces from database into memory"""
    global known_faces, known_ids, known_usernames
    try:
        known_faces.clear()
        known_ids.clear()
        known_usernames.clear()
        page_size = 1000
        offset = 0
        while True:
            cursor.execute("""
                SELECT user_id, username, face_encoding 
                FROM users 
                ORDER BY user_id 
                OFFSET ? ROWS 
                FETCH NEXT ? ROWS ONLY
            """, (offset, page_size))
            rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                user_id = row.user_id
                username = row.username
                encoding_str = row.face_encoding
                if encoding_str:
                    encoding = np.array([float(x) for x in encoding_str.split(',')])
                    known_faces.append(encoding)
                    known_ids.append(user_id)
                    known_usernames.append(username)
            offset += page_size
        print(f"Loaded {len(known_faces)} faces from database")
    except Exception as e:
        print(f"Error loading faces: {e}")

def is_face_enrolled(face_encoding):
    """
    Checks if a face is already enrolled in the system
    Returns user_id and username if found, None if not found
    """
    if not known_faces:
        return None, None
    face_encoding = np.array(face_encoding)
    distances = np.linalg.norm(np.array(known_faces) - face_encoding, axis=1)
    matches = distances < 0.6
    if np.any(matches):
        match_index = np.where(matches)[0][0]
        return known_ids[match_index], known_usernames[match_index]
    return None, None

def delete_user(username):
    """
    Deletes a user from the system
    Checks if user exists and handles related records
    """
    try:
        # Check if user exists
        cursor.execute("SELECT user_id, email FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if not result:
            print(f"Username {username} not found")
            return
            
        user_id, email = result
        
        # Check if user is already deleted
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        if cursor.fetchone()[0] == 0:
            print(f"User {username} has already been deleted")
            return
            
        # Delete user's records
        cursor.execute("DELETE FROM login_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM workspace_access_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        
        # Delete user's QR code file if it exists
        qr_files = [f for f in os.listdir("qr_codes") if f.startswith(f"{username}_")]
        for qr_file in qr_files:
            try:
                os.remove(os.path.join("qr_codes", qr_file))
            except:
                pass
                
        print(f"User {username} (ID: {user_id}, Email: {email}) deleted successfully")
        load_known_faces()
    except Exception as e:
        print(f"Error deleting user: {e}")

def capture_image():
    """
    Captures an image from the webcam
    Returns the captured frame and temporary filename
    """
    cap = None
    for index in range(3):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully opened camera at index {index}")
                    break
                cap.release()
        except:
            cap.release()
    else:
        print("No working camera found")
        return None, None

    countdown = 5
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
        cv2.putText(frame, f"Position face, capturing in {countdown}...", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(face_locations) > 1:
            cv2.putText(frame, f"Multiple faces ({len(face_locations)})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam - Position Your Face', frame)
        elapsed = time.time() - start_time
        new_countdown = 5 - int(elapsed)
        if new_countdown != countdown:
            countdown = new_countdown
            if countdown <= 0:
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None, None
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    if ret:
        temp_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_filename, frame)
        return frame, temp_filename
    return None, None

def encode_image_to_base64(filename):
    """Converts an image file to base64 string"""
    try:
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def recognize_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    if len(face_locations) == 0:
        return None, None, "No face detected"
    if len(face_locations) > 1:
        return None, None, f"Multiple faces detected ({len(face_locations)})"
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if not face_encodings:
        return None, None, "Could not generate face encoding"
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.7)
    if True in matches:
        first_match_index = matches.index(True)
        return known_ids[first_match_index], known_usernames[first_match_index], f"Welcome back, {known_usernames[first_match_index]}!"
    return None, None, "Face not recognized"

def send_qr_code_email(username, email, qr_img):
    """
    Sends QR code via email using Gmail SMTP
    Requires Gmail App Password for authentication
    """
    # Email configuration
    sender_email = "makoflash05@gmail.com"  # Your Gmail address
    
    # To get an App Password:
    # 1. Go to your Google Account settings
    # 2. Enable 2-Step Verification if not already enabled
    # 3. Go to Security → App passwords
    # 4. Select "Mail" as the app and "Other" as the device
    # 5. Use the generated 16-character password here
    sender_password = "YOUR_APP_PASSWORD_HERE"  # Replace with your 16-character App Password
    
    subject = "Your QR Code for Login"
    body = f"Dear {username},\n\nAttached is your QR code for future logins.\n\nBest regards,\nYour Team"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Generate PDF
    pdf_filename = f"{username}_qr_code.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, f"QR Code for {username}")
    qr_img.save("temp_qr_for_pdf.png")
    c.drawImage("temp_qr_for_pdf.png", 100, 500, width=200, height=200)
    c.save()
    os.remove("temp_qr_for_pdf.png")

    # Attach PDF
    with open(pdf_filename, "rb") as pdf_file:
        pdf_attachment = MIMEBase('application', 'octet-stream')
        pdf_attachment.set_payload(pdf_file.read())
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header('Content-Disposition', f'attachment; filename={pdf_filename}')
        msg.attach(pdf_attachment)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        print(f"QR code sent to {email}")
    except Exception as e:
        print(f"Error sending email: {e}")
        print("\nTo fix email sending issues:")
        print("1. Go to your Google Account settings")
        print("2. Enable 2-Step Verification if not already enabled")
        print("3. Go to Security → App passwords")
        print("4. Select 'Mail' as the app and 'Other' as the device")
        print("5. Use the generated 16-character password in the code")
    finally:
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)

def enroll_user(username, email):
    """
    Enrolls a new user into the system
    Captures face, generates QR code, and saves to database
    """
    # Validate email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        print("Invalid email format. Please enter a valid email address.")
        return

    cursor.execute("SELECT user_id FROM users WHERE username = ? OR email = ?", (username, email))
    existing_user = cursor.fetchone()
    if existing_user:
        print(f"Username or email already enrolled")
        return

    image, temp_filename = capture_image()
    if image is None or temp_filename is None:
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) != 1:
        print("Please ensure only one face is in the image")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    existing_user_id, existing_username = is_face_enrolled(face_encoding)
    if existing_user_id:
        print(f"Face already enrolled under {existing_username}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    user_id, error = generate_user_id(username)
    if error:
        print(error)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    encoding_str = ','.join(map(str, face_encoding))
    qr_data = f"{user_id}|{username}"  # Reduced QR data size
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=5
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill='black', back_color='white')
    
    # Create qr_codes directory if it doesn't exist
    os.makedirs("qr_codes", exist_ok=True)
    
    # Save QR code with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qr_filename = f"qr_codes/{username}_{timestamp}_qr.png"
    qr_img.save(qr_filename)
    
    print(f"\nQR Code has been saved to: {os.path.abspath(qr_filename)}")
    print("Please save this QR code for future logins.")

    # Save QR code for database
    qr_img.save("temp_qr.png")
    with open("temp_qr.png", "rb") as qr_file:
        qr_base64 = base64.b64encode(qr_file.read()).decode('utf-8')
    os.remove("temp_qr.png")

    image_base64 = encode_image_to_base64(temp_filename)
    if image_base64 is None:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    try:
        cursor.execute(
            "INSERT INTO users (user_id, username, email, face_encoding, qr_code_base64) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, email, encoding_str, qr_base64)
        )
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, 1, image_base64, format_timestamp())
        )
        conn.commit()
        print(f"User {username} enrolled successfully with ID: {user_id}")
        print(f"Email {email} has been stored in the database")
        load_known_faces()
        
        # Ask if user wants to receive QR code via email
        print("\nWould you like to receive the QR code via email? (yes/no)")
        email_choice = input("Enter your choice: ").strip().lower()
        if email_choice == 'yes':
            send_qr_code_email(username, email, qr_img)
    except Exception as e:
        print(f"Error enrolling user: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def login(username):
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if not cursor.fetchone():
        print(f"Username '{username}' not registered")
        return

    print("Choose login method: (1) Face Recognition, (2) QR Code")
    method = input("Enter choice (1-2): ").strip()
    
    image, temp_filename = capture_image()
    if image is None or temp_filename is None:
        return

    user_id = None
    recognized_username = None
    message = None

    if method == '1':
        user_id, recognized_username, message = recognize_face(image)
    elif method == '2':
        decoded_objects = decode(Image.open(temp_filename))
        if not decoded_objects:
            print("No QR code found")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return
        qr_data = decoded_objects[0].data.decode('utf-8')
        try:
            qr_user_id, qr_username = qr_data.split('|')
            cursor.execute("SELECT user_id, username FROM users WHERE user_id = ?", (qr_user_id,))
            result = cursor.fetchone()
            if result and result[1] == username:
                user_id = qr_user_id
                recognized_username = qr_username
                message = f"Welcome back, {recognized_username}!"
            else:
                message = "QR code does not match username"
        except:
            message = "Invalid QR code format"
    else:
        print("Invalid method")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    image_base64 = encode_image_to_base64(temp_filename)
    success = 1 if user_id and recognized_username == username else 0

    try:
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, recognized_username if user_id else username, success, image_base64, format_timestamp())
        )
        conn.commit()
        print(message)
        if success:
            workspace_url = "https://app.fabric.microsoft.com/groups/3572b987-13cf-4e12-8727-c88463aa3c0d/list?experience=fabric-developer&clientSideAuth=0"
            webbrowser.open(workspace_url)
            cursor.execute(
                "INSERT INTO workspace_access_logs (user_id, username, access_timestamp) "
                "VALUES (?, ?, ?)",
                (user_id, recognized_username, format_timestamp())
            )
            conn.commit()
            print(f"Accessing Fabric workspace for {recognized_username}")
    except Exception as e:
        print(f"Error logging attempt: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def logout(user_id):
    """
    Logs out a user and records the logout in the database
    Checks if user is already logged out
    """
    try:
        # Check if user exists
        cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if not result:
            print(f"User ID {user_id} not found")
            return
            
        username = result[0]
        
        # Check if user is already logged out
        cursor.execute("""
            SELECT TOP 1 is_logout 
            FROM login_logs 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        """, (user_id,))
        last_logout = cursor.fetchone()
        
        if last_logout and last_logout[0] == 1:
            print(f"User {username} is already logged out")
            return
            
        # Record logout
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, is_logout, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, 1, 1, format_timestamp())
        )
        conn.commit()
        print(f"User {username} has been logged out successfully")
    except Exception as e:
        print(f"Error logging out: {e}")

def main():
    load_known_faces()
    while True:
        print("\nOptions: (1) Enroll, (2) Login, (3) Logout, (4) Delete User, (5) Exit")
        choice = input("Enter choice (1-5): ").strip()
        if choice == '1':
            username = input("Enter username: ").strip()
            email = input("Enter email: ").strip()
            if username and email:
                enroll_user(username, email)
            else:
                print("Username and email cannot be empty")
        elif choice == '2':
            username = input("Enter username: ").strip()
            if username:
                login(username)
            else:
                print("Username cannot be empty")
        elif choice == '3':
            user_id = input("Enter user ID (e.g., Michael_1): ").strip()
            if user_id:
                logout(user_id)
            else:
                print("User ID cannot be empty")
        elif choice == '4':
            username = input("Enter username to delete: ").strip()
            if username:
                delete_user(username)
            else:
                print("Username cannot be empty")
        elif choice == '5':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()

cursor.close()
conn.close()