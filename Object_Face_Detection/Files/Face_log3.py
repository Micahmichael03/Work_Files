# Import required libraries
import cv2  # For webcam capture and image processing
import face_recognition  # For face detection and recognition
import pyodbc  # For connecting to Fabric SQL database
import numpy as np  # For handling facial encoding arrays
import base64  # For encoding images as base64 strings
import os  # For file operations (e.g., deleting temporary files)
import webbrowser  # For opening the Fabric workspace in a browser
import time  # For implementing a countdown timer
import re  # For validating usernames
from datetime import datetime  # For timestamping files and logs

# Database connection with hardcoded Azure AD credentials
connection_string = (
    "Driver={ODBC Driver 18 for SQL Server};"  # Specify ODBC Driver 18
    "Server=gjmfi7jmo2delewe55pp7ledge-q64xenopcmje5bzhzccghkr4bu.database.fabric.microsoft.com,1433;"  # Fabric SQL server
    "Database=warehouseDB-5606c843-5230-4432-9741-392553ea9fd5;"  # Database name
    "Encrypt=yes;"  # Enable encryption
    "TrustServerCertificate=no;"  # Verify server certificate
    "Authentication=ActiveDirectoryPassword;"  # Azure AD authentication
    "UID=Micahmichael@makoflash02gmail.onmicrosoft.com;"  # Azure AD username
    "PWD=@Chukwuemeka2025"  # Azure AD password
)

# Attempt to connect to the database
try:
    conn = pyodbc.connect(connection_string)  # Establish connection
    cursor = conn.cursor()  # Create cursor for SQL queries
except Exception as e:
    print(f"Database connection error: {e}")  # Print error if connection fails
    exit()  # Exit on failure

# Initialize global lists for known face data
known_faces = []  # Store facial encodings
known_ids = []  # Store user IDs (e.g., Michael_1)
known_usernames = []  # Store usernames

# Function to format timestamp as MM/DD/YYYY HH:MM AM/PM
def format_timestamp():
    return datetime.now().strftime("%m/%d/%Y %I:%M %p")  # e.g., 05/22/2025 10:41 AM

# Function to generate user_id as username_number
def generate_user_id(username):
    """
    Generate a unique user ID with better handling for large numbers
    """
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return None, "Username can only contain letters, numbers, or underscores"
    
    try:
        # Get the highest number for this username prefix
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

# Function to load known faces from the database
def load_known_faces():
    """
    Load known faces from the database with pagination for large datasets
    """
    global known_faces, known_ids, known_usernames
    try:
        # Clear existing data
        known_faces.clear()
        known_ids.clear()
        known_usernames.clear()

        # Use pagination to load faces in chunks
        page_size = 1000  # Number of faces to load at once
        offset = 0
        
        while True:
            # Query users table with pagination
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
                    # Convert string to numpy array
                    encoding = np.array([float(x) for x in encoding_str.split(',')])
                    known_faces.append(encoding)
                    known_ids.append(user_id)
                    known_usernames.append(username)
            
            offset += page_size
            
        print(f"Loaded {len(known_faces)} faces from database")
    except Exception as e:
        print(f"Error loading faces: {e}")

# Function to check if face is already enrolled
def is_face_enrolled(face_encoding):
    """
    Check if a face is already enrolled using efficient comparison
    """
    if not known_faces:
        return None, None

    # Use numpy for efficient comparison
    face_encoding = np.array(face_encoding)
    distances = np.linalg.norm(np.array(known_faces) - face_encoding, axis=1)
    matches = distances < 0.6  # Tolerance threshold

    if np.any(matches):
        match_index = np.where(matches)[0][0]
        return known_ids[match_index], known_usernames[match_index]
    return None, None

# Function to delete a user
def delete_user(username):
    try:
        # Check if username exists
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if not result:
            print(f"Username {username} not found")
            return
        user_id = result[0]
        # Delete from related tables first
        cursor.execute("DELETE FROM login_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM workspace_access_logs WHERE user_id = ?", (user_id,))
        # Delete from users table
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()  # Commit transaction
        print(f"User {username} (ID: {user_id}) deleted successfully")
        load_known_faces()  # Reload faces
    except Exception as e:
        print(f"Error deleting user: {e}")  # Print error if query fails

# Function to capture an image from the webcam with a countdown
def capture_image():
    # Try camera indices (0-2) with DirectShow backend
    cap = None
    for index in range(3):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow
            if cap.isOpened():
                ret, test_frame = cap.read()  # Test frame capture
                if ret and test_frame is not None:
                    print(f"Successfully opened camera at index {index}")
                    break
                cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"Error trying camera index {index}: {e}")
            if cap is not None:
                cap.release()
    else:
        print("No working camera found. Please check connection and permissions.")
        return None, None

    # Initialize countdown (5 seconds)
    countdown = 5
    start_time = time.time()

    # Display webcam feed with countdown and face feedback
    while True:
        ret, frame = cap.read()  # Read frame
        if not ret:
            print("Failed to capture frame")
            cap.release()
            return None, None

        # Add countdown text
        cv2.putText(
            frame,
            f"Position face, capturing in {countdown}...",
            (10, 30),  # Text position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Scale
            (0, 255, 0),  # Green
            2  # Thickness
        )

        # Add face detection feedback
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(face_locations) > 1:
            cv2.putText(frame, f"Multiple faces detected ({len(face_locations)})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam - Position Your Face', frame)  # Display frame

        # Update countdown
        elapsed = time.time() - start_time
        new_countdown = 5 - int(elapsed)
        if new_countdown != countdown:
            countdown = new_countdown
            if countdown <= 0:
                break

        # Check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None, None

    # Capture final image
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if ret:
        # Generate unique filename
        temp_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_filename, frame)
        return frame, temp_filename
    print("Failed to capture image")
    return None, None

# Function to encode an image as base64
def encode_image_to_base64(filename):
    try:
        with open(filename, "rb") as image_file:  # Open file
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')  # Encode to base64
        return encoded_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# Function to recognize a face
def recognize_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face_locations = face_recognition.face_locations(rgb_image, model="hog")  # Detect faces with HOG
    if len(face_locations) == 0:
        return None, None, "No face detected. Please ensure your face is clearly visible."
    if len(face_locations) > 1:
        return None, None, f"Multiple faces detected ({len(face_locations)}). Please ensure one face."
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)  # Compute encodings
    if not face_encodings:
        return None, None, "Could not generate face encoding. Try better lighting."
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.7)  # Compare faces
    if True in matches:
        first_match_index = matches.index(True)
        user_id = known_ids[first_match_index]
        username = known_usernames[first_match_index]
        return user_id, username, f"Welcome back, {username}!"
    return None, None, "Face not recognized. Please ensure you are enrolled."

# Function to enroll a new user
def enroll_user(username):
    # Check if username exists
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        print(f"Username {username} already enrolled")
        return

    # Capture image
    image, temp_filename = capture_image()
    if image is None or temp_filename is None:
        print("Failed to capture image")
        return

    # Detect face
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) != 1:
        print("Please ensure only one face is in the image")
        return

    # Compute face encoding
    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    
    # Check if face is already enrolled
    existing_user_id, existing_username = is_face_enrolled(face_encoding)
    if existing_user_id:
        print(f"Face already enrolled under username {existing_username}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    # Generate user_id
    user_id, error = generate_user_id(username)
    if error:
        print(error)
        return

    # Convert encoding to string
    encoding_str = ','.join(map(str, face_encoding))
    
    # Encode image
    image_base64 = encode_image_to_base64(temp_filename)
    if image_base64 is None:
        print("Failed to process image")
        return

    try:
        # Insert user
        cursor.execute(
            "INSERT INTO users (user_id, username, face_encoding) VALUES (?, ?, ?)",
            (user_id, username, encoding_str)
        )
        # Insert login log
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, 1, image_base64, format_timestamp())
        )
        conn.commit()
        print(f"User {username} enrolled successfully with ID: {user_id}")
        load_known_faces()  # Reload faces
    except Exception as e:
        print(f"Error enrolling user: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Function to perform login
def login(username):
    # First check if the username exists in the database
    try:
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if not result:
            print(f"Username '{username}' is not registered in the system.")
            return
    except Exception as e:
        print(f"Error checking username: {e}")
        return

    # If username exists, proceed with face capture
    image, temp_filename = capture_image()
    if image is None or temp_filename is None:
        print("Failed to capture image")
        return

    # Perform face recognition
    user_id, recognized_username, message = recognize_face(image)
    image_base64 = encode_image_to_base64(temp_filename)
    success = 1 if user_id else 0

    try:
        # Log the login attempt
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, recognized_username, success, image_base64, format_timestamp())
        )
        conn.commit()
        print(message)
        
        # If login is successful and username matches
        if user_id and recognized_username == username:
            # Open Fabric workspace URL
            workspace_url = "https://app.fabric.microsoft.com/groups/3572b987-13cf-4e12-8727-c88463aa3c0d/list?experience=fabric-developer&clientSideAuth=0"
            webbrowser.open(workspace_url)
            
            # Log workspace access
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
        # Clean up temporary image file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Function to perform logout
def logout(user_id):
    try:
        # Verify user_id exists
        cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if not result:
            print(f"User ID {user_id} not found")
            return
        username = result[0]
        # Insert logout log
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, is_logout, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, 1, 1, format_timestamp())
        )
        conn.commit()
        print("Logout successful")
    except Exception as e:
        print(f"Error logging out: {e}")

# Main function
def main():
    load_known_faces()
    while True:
        print("\nOptions: (1) Enroll, (2) Login, (3) Logout, (4) Delete User, (5) Exit")
        choice = input("Enter choice (1-5): ").strip()
        if choice == '1':
            username = input("Enter username: ").strip()
            if username:
                enroll_user(username)
            else:
                print("Username cannot be empty")
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

# Run main function
if __name__ == "__main__":
    main()

# Close database connection
cursor.close()
conn.close()