import re
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import base64
from PIL import Image
from pyzbar.pyzbar import decode
import qrcode
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from class_config import draw_bounding_box, get_color

def capture_image(prompt="Press 'c' to capture image, 'q' to quit", show_bounding_boxes=True):
    """
    Captures an image from the webcam with optional bounding boxes
    Returns the captured frame and temporary filename
    """
    cap = None
    for index in range(3):  # Try different camera indices
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
    
    print(prompt)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
            
        if show_bounding_boxes:
            # Process frame for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for (top, right, bottom, left) in face_locations:
                frame = draw_bounding_box(frame, (left, top, right, bottom), "Face", color=(0, 255, 0))
            
            # Process frame for QR code detection
            decoded_objects = decode(Image.fromarray(rgb_frame))
            for obj in decoded_objects:
                x, y, w, h = obj.rect
                frame = draw_bounding_box(frame, (x, y, x+w, y+h), "QR Code", color=(255, 0, 0))
        
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            temp_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(temp_filename, frame)
            cap.release()
            cv2.destroyAllWindows()
            return frame, temp_filename
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None, None

def encode_image_to_base64(filename):
    """Converts an image file to base64 string"""
    try:
        with open(filename, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def format_timestamp():
    """Returns current timestamp in MM/DD/YYYY HH:MM AM/PM format"""
    return datetime.now().strftime("%m/%d/%Y %I:%M %p")

def generate_user_id(username, cursor):
    """
    Generates a unique user ID based on username
    Format: username_number (e.g., john_1, john_2)
    """
    if not re.match(r'^[a-zA-Z0-9_]+$', username):  # Validate username format
        return None, "Username can only contain letters, numbers, or underscores"
    try:
        # Get all existing usernames that match the pattern
        cursor.execute("SELECT username FROM users WHERE username LIKE ?", (f"{username}_%",))
        existing_usernames = [row[0] for row in cursor.fetchall()]
        
        # Find the highest number
        max_number = 0
        for existing_username in existing_usernames:
            try:
                number = int(existing_username.split('_')[-1])
                max_number = max(max_number, number)
            except (ValueError, IndexError):
                continue
        
        # Generate new user ID
        next_number = max_number + 1
        return f"{username}_{next_number}", None
    except Exception as e:
        return None, f"Error generating user_id: {e}"

def is_face_enrolled(face_encoding, known_faces, known_ids, known_usernames):
    """
    Checks if a face is already enrolled
    Returns user_id and username if found, None if not
    """
    if not known_faces:
        return None, None
    face_encoding = np.array(face_encoding)
    distances = np.linalg.norm(np.array(known_faces) - face_encoding, axis=1)  # Compute distances to known faces
    matches = distances < 0.6  # Threshold for face match
    if np.any(matches):
        match_index = np.where(matches)[0][0]
        return known_ids[match_index], known_usernames[match_index]
    return None, None

def complete_enrollment(user_id, username, email, face_encoding, image, temp_filename, cursor, conn):
    """Complete the enrollment process"""
    encoding_str = ','.join(map(str, face_encoding))
    qr_data = f"{user_id}|{username}"
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill='black', back_color='white')
    
    os.makedirs("qr_codes", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qr_filename = f"qr_codes/{username}_{timestamp}_qr.png"
    qr_img.save(qr_filename)
    print(f"\nQR Code has been saved to: {os.path.abspath(qr_filename)}")
    
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
            "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, 1, image_base64, format_timestamp())
        )
        conn.commit()
        print(f"User {username} enrolled successfully with ID: {user_id}")
        print(f"Email {email} has been stored in the database")
        
        print("\nWould you like to receive the QR code via email? (yes/no)")
        email_choice = input("Enter your choice: ").strip().lower()
        if email_choice == 'yes':
            send_qr_code_email(username, email, qr_img)
    except Exception as e:
        print(f"Error enrolling user: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def send_qr_code_email(username, email, qr_img):
    """
    Sends QR code via email using Gmail SMTP
    Requires Gmail App Password for authentication
    """
    sender_email = os.getenv('GMAIL_USER', 'makoflash05@gmail.com')
    sender_password = os.getenv('GMAIL_APP_PASSWORD')  # Get from environment variable
    
    if not sender_password:
        print("Error: Gmail App Password not found in environment variables")
        print("\nTo fix email sending issues:")
        print("1. Set the GMAIL_APP_PASSWORD environment variable with your Gmail App Password")
        print("2. Go to your Google Account settings")
        print("3. Enable 2-Step Verification if not already enabled")
        print("4. Go to Security → App passwords")
        print("5. Select 'Mail' as the app and 'Other' as the device")
        print("6. Use the generated 16-character password")
        return
        
    subject = "Your QR Code for Login"
    body = f"Dear {username},\n\nAttached is your QR code for future logins.\n\nBest regards,\nYour Team"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    pdf_filename = f"{username}_qr_code.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, f"QR Code for {username}")
    qr_img.save("temp_qr_for_pdf.png")
    c.drawImage("temp_qr_for_pdf.png", 100, 500, width=200, height=200)
    c.save()
    os.remove("temp_qr_for_pdf.png")
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
    finally:
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)

def load_known_faces(cursor):
    """Loads all known faces from database into memory"""
    known_faces = []
    known_ids = []
    known_usernames = []
    try:
        cursor.execute("SELECT user_id, username, face_encoding FROM users")
        rows = cursor.fetchall()
        for row in rows:
            user_id = row[0]
            username = row[1]
            encoding_str = row[2]
            if encoding_str:
                try:
                    encoding = np.array([float(x) for x in encoding_str.split(',')])
                    known_faces.append(encoding)
                    known_ids.append(user_id)
                    known_usernames.append(username)
                except (ValueError, TypeError) as e:
                    print(f"Error processing face encoding for user {username}: {e}")
                    continue
        print(f"Loaded {len(known_faces)} faces from database")
        return known_faces, known_ids, known_usernames
    except Exception as e:
        print(f"Error loading faces: {e}")
        return [], [], []

def process_login_image(image, temp_filename, username, cursor, conn, known_faces, known_ids, known_usernames):
    """Process login from a single image"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        print("No face detected in the image")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return None, None
    
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
    
    if True in matches:
        first_match_index = matches.index(True)
        matched_username = known_usernames[first_match_index]
        if matched_username == username:
            image_base64 = encode_image_to_base64(temp_filename)
            if image_base64:
                cursor.execute(
                    "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (known_ids[first_match_index], username, 1, image_base64, format_timestamp())
                )
                conn.commit()
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return known_ids[first_match_index], username
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return None, None

def process_login_qr(image, temp_filename, username, cursor, conn):
    """Process login from a QR code image"""
    decoded_objects = decode(Image.open(temp_filename))
    if not decoded_objects:
        print("No QR code found")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return None, None
    
    try:
        qr_data = decoded_objects[0].data.decode('utf-8')
        qr_user_id, qr_username = qr_data.split('|')
        if qr_username == username:
            image_base64 = encode_image_to_base64(temp_filename)
            if image_base64:
                cursor.execute(
                    "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (qr_user_id, username, 1, image_base64, format_timestamp())
                )
                conn.commit()
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return qr_user_id, username
    except:
        pass
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return None, None 