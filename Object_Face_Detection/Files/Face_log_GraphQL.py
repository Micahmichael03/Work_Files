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
import requests  # For GraphQL API calls
from datetime import datetime  # For timestamping files and logs

# Database connection with hardcoded Azure AD credentials
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

# Azure Function GraphQL endpoint (replace with your deployed URL)
GRAPHQL_ENDPOINT = "https://3572b98713cf4e128727c88463aa3c0d.z35.graphql.fabric.microsoft.com/v1/workspaces/3572b987-13cf-4e12-8727-c88463aa3c0d/graphqlapis/c840d0a8-af71-4c21-8264-90ba83126ccd/graphql"

# Attempt to connect to the database
try:
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
except Exception as e:
    print(f"Database connection error: {e}")
    exit()

# Initialize global lists for known face data
known_faces = []
known_ids = []
known_usernames = []

# Function to format timestamp as MM/DD/YYYY HH:MM AM/PM
def format_timestamp():
    return datetime.now().strftime("%m/%d/%Y %I:%M %p")

# Function to send GraphQL requests
def send_graphql_request(query, variables=None):
    headers = {"Content-Type": "application/json"}
    payload = {"query": query, "variables": variables or {}}
    try:
        response = requests.post(GRAPHQL_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"GraphQL request error: {e}")
        return None

# Function to generate user_id as username_number
def generate_user_id(username):
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return None, "Username can only contain letters, numbers, or underscores"
    try:
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        count = cursor.fetchone()[0]
        return f"{username}_{count + 1}", None
    except Exception as e:
        return None, f"Error generating user_id: {e}"

# Function to load known faces from the database
def load_known_faces():
    global known_faces, known_ids, known_usernames
    try:
        cursor.execute("SELECT user_id, username, face_encoding FROM users")
        known_faces.clear()
        known_ids.clear()
        known_usernames.clear()
        for row in cursor:
            user_id = row.user_id
            username = row.username
            encoding_str = row.face_encoding
            if encoding_str:
                encoding = np.array([float(x) for x in encoding_str.split(',')])
                known_faces.append(encoding)
                known_ids.append(user_id)
                known_usernames.append(username)
    except Exception as e:
        print(f"Error loading faces: {e}")

# Function to check if face is already enrolled
def is_face_enrolled(face_encoding):
    if not known_faces:
        return None, None
    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.7)
    if True in matches:
        index = matches.index(True)
        return known_ids[index], known_usernames[index]
    return None, None

# Function to delete a user
def delete_user(username):
    try:
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if not result:
            print(f"Username {username} not found")
            return
        user_id = result[0]
        cursor.execute("DELETE FROM login_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM workspace_access_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        print(f"User {username} (ID: {user_id}) deleted successfully")
        load_known_faces()
    except Exception as e:
        print(f"Error deleting user: {e}")

# Function to capture an image from the webcam with a countdown
def capture_image():
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
            else:
                cap.release()
        except Exception as e:
            print(f"Error trying camera index {index}: {e}")
            if cap is not None:
                cap.release()
    else:
        print("No working camera found. Please check connection and permissions.")
        return None, None

    countdown = 5
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            cap.release()
            return None, None

        cv2.putText(
            frame,
            f"Position face, capturing in {countdown}...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(face_locations) > 1:
            cv2.putText(frame, f"Multiple faces detected ({len(face_locations)})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
    print("Failed to capture image")
    return None, None

# Function to encode an image as base64
def encode_image_to_base64(filename):
    try:
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# Function to recognize a face
def recognize_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    if len(face_locations) == 0:
        return None, None, "No face detected. Please ensure your face is clearly visible."
    if len(face_locations) > 1:
        return None, None, f"Multiple faces detected ({len(face_locations)}). Please ensure one face."
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if not face_encodings:
        return None, None, "Could not generate face encoding. Try better lighting."
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.7)
    if True in matches:
        first_match_index = matches.index(True)
        user_id = known_ids[first_match_index]
        username = known_usernames[first_match_index]
        return user_id, username, f"Welcome back, {username}!"
    return None, None, "Face not recognized. Please ensure you are enrolled."

# Function to enroll a new user
def enroll_user(username):
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        print(f"Username {username} already enrolled")
        return

    image, temp_filename = capture_image()
    if image is None or temp_filename is None:
        print("Failed to capture image")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) != 1:
        print("Please ensure only one face is in the image")
        return

    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    existing_user_id, existing_username = is_face_enrolled(face_encoding)
    if existing_user_id:
        print(f"Face already enrolled under username {existing_username}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return

    user_id, error = generate_user_id(username)
    if error:
        print(error)
        return

    encoding_str = ','.join(map(str, face_encoding))
    image_base64 = encode_image_to_base64(temp_filename)
    if image_base64 is None:
        print("Failed to process image")
        return

    try:
        cursor.execute(
            "INSERT INTO users (user_id, username, face_encoding) VALUES (?, ?, ?)",
            (user_id, username, encoding_str)
        )
        cursor.execute(
            "INSERT INTO login_logs (user_id, username, success, image_base64, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, 1, image_base64, format_timestamp())
        )
        conn.commit()
        print(f"User {username} enrolled successfully with ID: {user_id}")
        load_known_faces()
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
            
            # Log workspace access via GraphQL
            mutation = """
            mutation($user_id: String!, $username: String!, $access_timestamp: String!) {
              logWorkspaceAccess(user_id: $user_id, username: $username, access_timestamp: $access_timestamp) {
                access_id
              }
            }
            """
            response = send_graphql_request(mutation, {
                "user_id": user_id,
                "username": recognized_username,
                "access_timestamp": format_timestamp()
            })
            if response and 'data' in response and response['data']['logWorkspaceAccess']:
                print(f"Accessing Fabric workspace for {recognized_username} (logged via GraphQL)")
            else:
                print("Error logging workspace access via GraphQL")
    except Exception as e:
        print(f"Error logging attempt: {e}")
    finally:
        # Clean up temporary image file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Function to perform logout
def logout(user_id):
    try:
        cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if not result:
            print(f"User ID {user_id} not found")
            return
        username = result[0]
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