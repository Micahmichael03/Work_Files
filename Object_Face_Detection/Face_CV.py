import cv2  # OpenCV for image and video processing
import face_recognition  # For face detection and recognition
import pyodbc  # For SQL Server database connection
import sqlite3  # For SQLite database connection
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
import smtplib  # For sending emails
from email.mime.multipart import MIMEMultipart  # For email structure
from email.mime.text import MIMEText  # For email text content
from email.mime.base import MIMEBase  # For email attachments
from email import encoders  # For encoding email attachments
from reportlab.pdfgen import canvas  # For generating PDF reports
from reportlab.lib.pagesizes import letter  # For PDF page size
from ultralytics import YOLO  # For object detection
import uuid  # For generating unique IDs
import json  # For JSON operations
from io import BytesIO  # For in-memory file operations
import requests  # For making HTTP requests
from video_processor import VideoProcessor
from class_config import draw_bounding_box, get_color
from common_functions import (
    capture_image,
    complete_enrollment,
    generate_user_id,
    is_face_enrolled,
    format_timestamp,
    encode_image_to_base64,
    send_qr_code_email,
    load_known_faces,
    process_login_image,
    process_login_qr
)
from enhanced_functions import (
    enroll_user,
    login,
    delete_user,
    process_face_login,
    process_qr_login,
    verify_face_deletion,
    verify_qr_deletion
)

# Global variables for database connection and face data
conn = None
cursor = None
known_faces = []  # Stores face encodings
known_ids = []    # Stores corresponding user IDs
known_usernames = []  # Stores corresponding usernames

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

def load_known_faces(cursor):
    """Loads all known faces from database into memory"""
    global known_faces, known_ids, known_usernames
    try:
        known_faces.clear()  # Clear existing face data
        known_ids.clear()
        known_usernames.clear()
        
        # Use SQLite-compatible query
        cursor.execute("""
            SELECT user_id, username, face_encoding 
            FROM users 
            ORDER BY user_id 
        """)
        rows = cursor.fetchall()
        
        for row in rows:
            user_id = row[0]
            username = row[1]
            encoding_str = row[2]
            if encoding_str:
                try:
                    encoding = np.array([float(x) for x in encoding_str.split(',')])  # Convert encoding string to array
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

def is_face_enrolled(face_encoding):
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

def delete_user(username, cursor, conn, known_faces, known_ids, known_usernames):
    """Deletes a user with video verification"""
    try:
        cursor.execute("SELECT user_id, email FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if not result:
            print(f"Username {username} not found")
            return
        
        user_id, email = result
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        if cursor.fetchone()[0] == 0:
            print(f"User {username} has already been deleted")
            return
        
        print("\nChoose verification method:")
        print("1. Face Recognition (Image)")
        print("2. Face Recognition (Video)")
        print("3. QR Code (Image)")
        print("4. QR Code (Video)")
        method = input("Enter choice (1-4): ").strip()
        
        if method in ['1', '2']:
            if not verify_face_deletion(username, method == '2', known_faces, known_ids, known_usernames):
                print("Face verification failed")
                return
        elif method in ['3', '4']:
            if not verify_qr_deletion(username, method == '4'):
                print("QR code verification failed")
                return
        else:
            print("Invalid method")
            return
            
        try:
            # Delete user's data from all related tables
            cursor.execute("DELETE FROM login_logs WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM workspace_access_logs WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            
            # Delete user's QR code files
            qr_files = [f for f in os.listdir("qr_codes") if f.startswith(f"{username}_")]
            for qr_file in qr_files:
                try:
                    os.remove(os.path.join("qr_codes", qr_file))
                except:
                    pass
                    
            print(f"User {username} (ID: {user_id}, Email: {email}) deleted successfully")
            # Reload known faces after deletion
            load_known_faces(cursor)
        except Exception as e:
            print(f"Error deleting user: {e}")
            conn.rollback()
    except Exception as e:
        print(f"Error in delete_user: {e}")

def verify_face_deletion(username, use_video=False, known_faces=None, known_ids=None, known_usernames=None):
    """Verify user identity for deletion using face recognition"""
    if use_video:
        video_processor = VideoProcessor()
        print("Press 'c' to capture frame, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return False
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
                if True in matches:
                    first_match_index = matches.index(True)
                    matched_username = known_usernames[first_match_index]
                    if matched_username == username:
                        frame = draw_bounding_box(frame, (left, top, right, bottom), f"Face: {username}", color=(0, 255, 0))
                        cv2.imshow('Verification', frame)
                        cv2.waitKey(1000)
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                    else:
                        frame = draw_bounding_box(frame, (left, top, right, bottom), "Face: Unknown", color=(255, 0, 0))
                else:
                    frame = draw_bounding_box(frame, (left, top, right, bottom), "Face: Unknown", color=(255, 0, 0))
            
            cv2.imshow('Verification', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return False
    else:
        image, temp_filename = capture_image("Position face for verification, press 'c' to capture, 'q' to quit")
        if image is None or temp_filename is None:
            return False
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False
            
        matches = face_recognition.compare_faces(known_faces, face_encodings[0], tolerance=0.6)
        if True in matches:
            first_match_index = matches.index(True)
            if known_usernames[first_match_index] == username:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                return True
                
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return False

def verify_qr_deletion(username, use_video=False):
    """Verify user identity for deletion using QR code"""
    if use_video:
        video_processor = VideoProcessor()
        print("Press 'c' to capture frame, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return False
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            decoded_objects = decode(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            for obj in decoded_objects:
                x, y, w, h = obj.rect
                frame = draw_bounding_box(frame, (x, y, x+w, y+h), "QR Code", color=(255, 0, 0))
                
                try:
                    qr_data = obj.data.decode('utf-8')
                    qr_user_id, qr_username = qr_data.split('|')
                    if qr_username == username:
                        cv2.imshow('Verification', frame)
                        cv2.waitKey(1000)
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                except:
                    continue
            
            cv2.imshow('Verification', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return False
    else:
        image, temp_filename = capture_image("Position QR code for verification, press 'c' to capture, 'q' to quit")
        if image is None or temp_filename is None:
            return False
            
        decoded_objects = decode(Image.open(temp_filename))
        if not decoded_objects:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False
            
        try:
            qr_data = decoded_objects[0].data.decode('utf-8')
            qr_user_id, qr_username = qr_data.split('|')
            if qr_username == username:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                return True
        except:
            pass
            
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return False

def recognize_face(image):
    """Recognizes a face in the image, returns user_id, username, and message"""
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

def enroll_user(username, email, cursor, conn, known_faces, known_ids, known_usernames):
    """
    Enrolls a new user into the system with options for image or video capture
    """
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        print("Invalid email format. Please enter a valid email address.")
        return
        
    cursor.execute("SELECT user_id FROM users WHERE username = ? OR email = ?", (username, email))
    existing_user = cursor.fetchone()
    if existing_user:
        print(f"Username or email already enrolled")
        return
        
    print("\nChoose capture method:")
    print("1. Single Image Capture")
    print("2. Video Stream")
    capture_choice = input("Enter choice (1-2): ").strip()
    
    if capture_choice == '1':
        image, temp_filename = capture_image("Position face, press 'c' to capture, 'q' to quit")
        if image is None or temp_filename is None:
            return
        process_enrollment_image(image, temp_filename, username, email, cursor, conn)
    elif capture_choice == '2':
        process_enrollment_video(username, email, cursor, conn)
    else:
        print("Invalid choice")

def process_enrollment_image(image, temp_filename, username, email, cursor, conn):
    """Process enrollment from a single image"""
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
        
    user_id, error = generate_user_id(username, cursor)
    if error:
        print(error)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return
        
    complete_enrollment(user_id, username, email, face_encoding, image, temp_filename, cursor, conn)

def process_enrollment_video(username, email, cursor, conn):
    """Process enrollment from video stream"""
    video_processor = VideoProcessor()
    print("Press 'c' to capture frame, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw bounding boxes
        for (top, right, bottom, left) in face_locations:
            frame = draw_bounding_box(frame, (left, top, right, bottom), "Face", color=(0, 255, 0))
            
        cv2.imshow('Enrollment', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(face_locations) == 1:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                existing_user_id, existing_username = is_face_enrolled(face_encoding)
                if existing_user_id:
                    print(f"Face already enrolled under {existing_username}")
                    break
                    
                user_id, error = generate_user_id(username, cursor)
                if error:
                    print(error)
                    break
                    
                temp_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(temp_filename, frame)
                complete_enrollment(user_id, username, email, face_encoding, frame, temp_filename, cursor, conn)
                break
            else:
                print("Please ensure only one face is visible")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def login(username, cursor, conn, known_faces, known_ids, known_usernames):
    """Logs in a user using face recognition or QR code with video support"""
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if not cursor.fetchone():
        print(f"Username '{username}' not registered")
        return None, None
        
    print("Choose login method:")
    print("1. Face Recognition (Image)")
    print("2. Face Recognition (Video)")
    print("3. QR Code (Image)")
    print("4. QR Code (Video)")
    method = input("Enter choice (1-4): ").strip()
    
    if method in ['1', '2']:
        return process_face_login(username, method == '2', cursor, conn, known_faces, known_ids, known_usernames)
    elif method in ['3', '4']:
        return process_qr_login(username, method == '4', cursor, conn)
    else:
        print("Invalid method")
        return None, None

def process_face_login(username, use_video=False, cursor=None, conn=None, known_faces=None, known_ids=None, known_usernames=None):
    """Process face-based login"""
    if use_video:
        video_processor = VideoProcessor()
        print("Press 'c' to capture frame, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return None, None
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Draw bounding boxes and check for matches
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
                if True in matches:
                    first_match_index = matches.index(True)
                    matched_username = known_usernames[first_match_index]
                    if matched_username == username:
                        frame = draw_bounding_box(frame, (left, top, right, bottom), f"Face: {username}", color=(0, 255, 0))
                        cv2.imshow('Login', frame)
                        cv2.waitKey(1000)  # Show the match for 1 second
                        cap.release()
                        cv2.destroyAllWindows()
                        return known_ids[first_match_index], username
                    else:
                        frame = draw_bounding_box(frame, (left, top, right, bottom), "Face: Unknown", color=(255, 0, 0))
                else:
                    frame = draw_bounding_box(frame, (left, top, right, bottom), "Face: Unknown", color=(255, 0, 0))
            
            cv2.imshow('Login', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return None, None
    else:
        image, temp_filename = capture_image("Position face, press 'c' to capture, 'q' to quit")
        if image is None or temp_filename is None:
            return None, None
        return process_login_image(image, temp_filename, username, cursor, conn, known_faces, known_ids, known_usernames)

def process_qr_login(username, use_video=False, cursor=None, conn=None):
    """Process QR code-based login"""
    if use_video:
        video_processor = VideoProcessor()
        print("Press 'c' to capture frame, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return None, None
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame for QR code detection
            decoded_objects = decode(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            for obj in decoded_objects:
                x, y, w, h = obj.rect
                frame = draw_bounding_box(frame, (x, y, x+w, y+h), "QR Code", color=(255, 0, 0))
                
                try:
                    qr_data = obj.data.decode('utf-8')
                    qr_user_id, qr_username = qr_data.split('|')
                    if qr_username == username:
                        cv2.imshow('Login', frame)
                        cv2.waitKey(1000)  # Show the match for 1 second
                        cap.release()
                        cv2.destroyAllWindows()
                        return qr_user_id, username
                except:
                    continue
            
            cv2.imshow('Login', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return None, None
    else:
        image, temp_filename = capture_image("Position QR code, press 'c' to capture, 'q' to quit")
        if image is None or temp_filename is None:
            return None, None
        return process_qr_login_image(image, temp_filename, username, cursor, conn)

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
                try:
                    # Get user_id from the database
                    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
                    user_id = cursor.fetchone()[0]
                    
                    # Check if user is already logged in
                    cursor.execute("""
                        SELECT TOP 1 is_logout 
                        FROM login_logs 
                        WHERE user_id = ? 
                        ORDER BY timestamp DESC
                    """, (user_id,))
                    last_log = cursor.fetchone()
                    
                    if last_log and last_log[0] == 0:
                        print(f"User {username} is already logged in")
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                        return None, None
                    
                    # Insert login log
                    cursor.execute("""
                        INSERT INTO login_logs (
                            user_id, 
                            username, 
                            success, 
                            image_base64, 
                            is_logout,
                            timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        username,
                        1,  # success
                        image_base64,
                        0,  # not a logout
                        format_timestamp()
                    ))
                    conn.commit()
                    print(f"Login logged successfully for user {username}")
                except Exception as e:
                    print(f"Error logging login: {e}")
                    conn.rollback()
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return known_ids[first_match_index], username
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        return None, None

def process_qr_login_image(image, temp_filename, username, cursor, conn):
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
                try:
                    # Check if user is already logged in
                    cursor.execute("""
                        SELECT TOP 1 is_logout 
                        FROM login_logs 
                        WHERE user_id = ? 
                        ORDER BY timestamp DESC
                    """, (qr_user_id,))
                    last_log = cursor.fetchone()
                    
                    if last_log and last_log[0] == 0:
                        print(f"User {username} is already logged in")
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                        return None, None
                    
                    # Insert login log
                    cursor.execute("""
                        INSERT INTO login_logs (
                            user_id, 
                            username, 
                            success, 
                            image_base64, 
                            is_logout,
                            timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        qr_user_id,
                        username,
                        1,  # success
                        image_base64,
                        0,  # not a logout
                        format_timestamp()
                    ))
                    conn.commit()
                    print(f"Login logged successfully for user {username}")
                except Exception as e:
                    print(f"Error logging login: {e}")
                    conn.rollback()
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return qr_user_id, username
    except Exception as e:
        print(f"Error processing QR code: {e}")
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return None, None

def logout(user_id):
    """Logs out a user and records the logout in the database"""
    try:
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
        
        # Insert logout log
        cursor.execute("""
            INSERT INTO login_logs (
                user_id, 
                username, 
                success, 
                image_base64,
                is_logout, 
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            username,
            1,  # success
            None,  # no image for logout
            1,  # is_logout
            format_timestamp()
        ))
        conn.commit()
        print(f"User {username} has been logged out successfully")
    except Exception as e:
        print(f"Error logging out: {e}")
        conn.rollback()

def order_points(pts):
    """
    Orders four points in the order: top-left, top-right, bottom-right, bottom-left
    Used for A4 paper detection in dimension estimation
    """
    pts = pts.reshape(4, 2)  # Reshape to 4x2 array
    pts = pts[np.argsort(pts[:, 1])]  # Sort by y-coordinate
    top = pts[:2]  # Top two points
    top = top[np.argsort(top[:, 0])]  # Sort by x-coordinate
    tl, tr = top[0], top[1]  # Top-left, top-right
    bottom = pts[2:]  # Bottom two points
    bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x-coordinate
    bl, br = bottom[0], bottom[1]  # Bottom-left, bottom-right
    return np.array([tl, tr, br, bl], dtype="float32")

def detect_a4_paper(image):
    """
    Detects an A4 paper in the image, returns its four corners
    Used for dimension estimation
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges with Canny
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter small contours
            rect = cv2.minAreaRect(contour)  # Get minimum area rectangle
            box = cv2.boxPoints(rect)  # Get rectangle corners
            box = np.int0(box)
            width = min(rect[1])  # Get rectangle width
            height = max(rect[1])  # Get rectangle height
            aspect_ratio = width / height
            if 0.6 < aspect_ratio < 0.8:  # A4 paper aspect ratio ~0.707
                return order_points(box)
    return None

def estimate_dimensions(image):
    """
    Estimates dimensions of a surface using an A4 paper as reference
    Returns width and height in meters
    """
    paper_corners = detect_a4_paper(image)
    if paper_corners is None:
        print("A4 paper not detected in the image")
        return None, None
    a4_mm = np.array([[0, 0], [297, 0], [297, 210], [0, 210]], dtype="float32")  # A4 dimensions in mm
    h, w = image.shape[:2]
    dst = np.array([[0, 0], [2970, 0], [2970, 2100], [0, 2100]], dtype="float32")  # Scaled A4 in pixels (10 pixels/mm)
    matrix = cv2.getPerspectiveTransform(paper_corners, dst)  # Compute homography matrix
    warped = cv2.warpPerspective(image, matrix, (2970, 2100))  # Transform to top-down view
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # Threshold to isolate non-black area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No surface area detected")
        return None, None
    largest_contour = max(contours, key=cv2.contourArea)  # Get largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding rectangle
    width_mm = w / 10  # Convert pixels to mm (10 pixels/mm)
    height_mm = h / 10
    width_m = width_mm / 1000  # Convert to meters
    height_m = height_mm / 1000
    return width_m, height_m

def get_cost_estimate(width_m, height_m, material_type):
    """
    Mock API function to estimate costs based on dimensions and material type
    Replace with actual API call when available
    """
    area = width_m * height_m
    material_cost = area * 100  # Example: $100 per square meter
    installation_cost = area * 150  # Example: $150 per square meter
    total_cost = material_cost + installation_cost
    return {
        "material_cost": round(material_cost, 2),
        "installation_cost": round(installation_cost, 2),
        "total_cost": round(total_cost, 2)
    }

def estimate_room(conn, cursor, user_id, username):
    """Estimates room dimensions and costs, logs to database"""
    image, temp_filename = capture_image("Place A4 paper on surface, press 'c' to capture, 'q' to quit")
    if image is None or temp_filename is None:
        return
    width_m, height_m = estimate_dimensions(image)
    if width_m is None or height_m is None:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return
    dimensions = f"{width_m:.2f}m x {height_m:.2f}m"
    material_type = input("Enter material type (e.g., ceramic_tile): ").strip()
    cost_data = get_cost_estimate(width_m, height_m, material_type)
    estimation_id = str(uuid.uuid4())
    timestamp = format_timestamp()
    image_base64 = encode_image_to_base64(temp_filename)
    if image_base64 is None:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return
    try:
        cursor.execute(
            "INSERT INTO estimations (estimation_id, user_id, image_base64, dimensions, material_type, material_cost, installation_cost, total_cost, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (estimation_id, user_id, image_base64, dimensions, material_type,
             cost_data["material_cost"], cost_data["installation_cost"], cost_data["total_cost"], timestamp)
        )
        conn.commit()
        print(f"Estimation saved successfully with ID: {estimation_id}")
        print(f"Estimation details: {dimensions}, Material: {material_type}, Total Cost: ${cost_data['total_cost']}")
    except Exception as e:
        print(f"Error saving estimation to database: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def detect_and_measure_object(image):
    """
    Detects objects in the image, measures their dimensions, and estimates costs
    Returns object information including dimensions, type, and cost estimates
    """
    # Load YOLO model
    model = YOLO("yolov8s.pt")
    
    # Convert image to RGB for YOLO
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform object detection with confidence threshold
    results = model(rgb_image, conf=0.25)  # Lower confidence threshold to detect more objects
    
    detected_objects = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get object class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf > 0.25:  # Lower confidence threshold
                # Get object name
                object_name = result.names[cls]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate dimensions using reference object (A4 paper)
                ref_corners = detect_a4_paper(image)
                if ref_corners is not None:
                    # Calculate pixel to mm ratio using A4 paper
                    a4_width_mm = 210  # A4 width in mm
                    a4_height_mm = 297  # A4 height in mm
                    
                    # Calculate pixel dimensions of A4 paper
                    a4_pixel_width = np.linalg.norm(ref_corners[1] - ref_corners[0])
                    a4_pixel_height = np.linalg.norm(ref_corners[2] - ref_corners[1])
                    
                    # Calculate pixel to mm ratios
                    pixel_to_mm_x = a4_width_mm / a4_pixel_width
                    pixel_to_mm_y = a4_height_mm / a4_pixel_height
                    
                    # Calculate object dimensions
                    obj_width_mm = (x2 - x1) * pixel_to_mm_x
                    obj_height_mm = (y2 - y1) * pixel_to_mm_y
                    
                    # Convert to meters
                    obj_width_m = obj_width_mm / 1000
                    obj_height_m = obj_height_mm / 1000
                    
                    # Get cost estimate
                    cost_data = get_cost_estimate(obj_width_m, obj_height_m, object_name)
                    
                    # Get e-commerce information
                    ecommerce_info = get_ecommerce_info(object_name)
                    
                    detected_objects.append({
                        'name': object_name,
                        'confidence': conf,
                        'dimensions': {
                            'width_m': obj_width_m,
                            'height_m': obj_height_m
                        },
                        'cost_estimate': cost_data,
                        'ecommerce_info': ecommerce_info
                    })
                else:
                    # If no A4 paper is detected, still add the object but with estimated dimensions
                    detected_objects.append({
                        'name': object_name,
                        'confidence': conf,
                        'dimensions': {
                            'width_m': 0.5,  # Default estimated dimensions
                            'height_m': 0.5
                        },
                        'cost_estimate': get_cost_estimate(0.5, 0.5, object_name),
                        'ecommerce_info': get_ecommerce_info(object_name)
                    })
    
    return detected_objects

def get_ecommerce_info(object_name):
    """
    Gets e-commerce information for the detected object
    Returns product information including price, website, and location
    """
    # This is a mock function - replace with actual API calls to e-commerce platforms
    return {
        'price_range': {
            'min': 50.0,
            'max': 200.0,
            'currency': 'USD'
        },
        'websites': [
            {
                'name': 'Amazon',
                'url': f'https://www.amazon.com/s?k={object_name.replace(" ", "+")}',
                'price': 100.0
            },
            {
                'name': 'eBay',
                'url': f'https://www.ebay.com/sch/i.html?_nkw={object_name.replace(" ", "+")}',
                'price': 75.0
            }
        ],
        'availability': 'In Stock',
        'shipping_info': 'Free shipping available'
    }

def generate_object_qr_code(object_info):
    """
    Generates a QR code containing object information
    Returns QR code image and base64 encoded data
    """
    # Create QR code data
    qr_data = {
        'object_name': object_info['name'],
        'dimensions': object_info['dimensions'],
        'cost_estimate': object_info['cost_estimate'],
        'ecommerce_info': object_info['ecommerce_info']
    }
    
    # Convert to JSON string
    qr_data_str = json.dumps(qr_data)
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data_str)
    qr.make(fit=True)
    
    # Create QR code image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffered = BytesIO()
    qr_img.save(buffered, format="PNG")
    qr_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return qr_img, qr_base64

def store_object_detection(conn, cursor, user_id, object_info, qr_base64):
    """Stores object detection results in the database"""
    try:
        # Generate unique ID for the detection
        detection_id = str(uuid.uuid4())
        
        # Store in database
        cursor.execute("""
            INSERT INTO object_detections (
                detection_id,
                user_id,
                object_name,
                dimensions,
                cost_estimate,
                ecommerce_info,
                qr_code_base64,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection_id,
            user_id,
            object_info['name'],
            json.dumps(object_info['dimensions']),
            json.dumps(object_info['cost_estimate']),
            json.dumps(object_info['ecommerce_info']),
            qr_base64,
            format_timestamp()
        ))
        
        conn.commit()
        print(f"Object detection stored successfully with ID: {detection_id}")
        return detection_id
    except Exception as e:
        print(f"Error storing object detection: {e}")
        return None

def detect_and_analyze_objects(conn, cursor, user_id, username, capture_image_func):
    """
    Main function for detecting and analyzing objects in a room
    """
    image, temp_filename = capture_image_func("Position camera to capture room/objects, press 'c' to capture, 'q' to quit")
    if image is None or temp_filename is None:
        return
    
    # Detect and measure objects
    detected_objects = detect_and_measure_object(image)
    
    if not detected_objects:
        print("No objects detected in the image")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return
    
    # Process each detected object
    for obj_info in detected_objects:
        # Generate QR code
        qr_img, qr_base64 = generate_object_qr_code(obj_info)
        
        # Store in database
        detection_id = store_object_detection(conn, cursor, user_id, obj_info, qr_base64)
        
        if detection_id:
            # Save QR code image
            os.makedirs("qr_codes", exist_ok=True)
            qr_filename = f"qr_codes/object_{detection_id}.png"
            qr_img.save(qr_filename)
            
            # Print information
            print(f"\nDetected: {obj_info['name']}")
            print(f"Dimensions: {obj_info['dimensions']['width_m']:.2f}m x {obj_info['dimensions']['height_m']:.2f}m")
            print(f"Estimated Cost: ${obj_info['cost_estimate']['total_cost']:.2f}")
            print(f"QR Code saved to: {os.path.abspath(qr_filename)}")
            
            # Open e-commerce website
            if obj_info['ecommerce_info']['websites']:
                webbrowser.open(obj_info['ecommerce_info']['websites'][0]['url'])
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

def process_video_stream(mode='object_detection', source=0, output_path=None):
    """
    Process video stream (webcam or file) with specified mode
    Args:
        mode: 'object_detection', 'face_recognition', or 'qr_code'
        source: Video source (0 for webcam, or path to video file)
        output_path: Path to save processed video (optional)
    """
    video_processor = VideoProcessor()
    
    if mode == 'face_recognition':
        # Load known faces for recognition
        load_known_faces(cursor)
        video_processor.load_known_faces(known_faces, known_ids, known_usernames)
    
    video_processor.process_video(source, mode, output_path)

def create_tables(cursor):
    """Create necessary database tables if they don't exist"""
    # Drop existing tables in reverse order to avoid foreign key constraints
    cursor.execute("DROP TABLE IF EXISTS login_logs")
    cursor.execute("DROP TABLE IF EXISTS workspace_access_logs")
    cursor.execute("DROP TABLE IF EXISTS users")
    
    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            user_id NVARCHAR(100) PRIMARY KEY,
            username NVARCHAR(50) NOT NULL,
            email VARCHAR(255),
            face_encoding TEXT,
            qr_code_base64 TEXT
        )
    """)
    
    # Create login_logs table
    cursor.execute("""
        CREATE TABLE login_logs (
            log_id INT PRIMARY KEY IDENTITY(1,1),
            user_id NVARCHAR(100) NULL,
            username NVARCHAR(50) NULL,
            success BIT,
            image_base64 TEXT,
            is_logout BIT DEFAULT 0,
            timestamp NVARCHAR(50),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # Create workspace_access_logs table
    cursor.execute("""
        CREATE TABLE workspace_access_logs (
            access_id INT PRIMARY KEY IDENTITY(1,1),
            user_id NVARCHAR(100) NULL,
            username NVARCHAR(50) NULL,
            access_timestamp NVARCHAR(50),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

def open_ms_fabric_workspace():
    """Opens the Microsoft Fabric workspace in the default browser"""
    workspace_url = "https://app.fabric.microsoft.com/home"
    webbrowser.open(workspace_url)
    print("Opening Microsoft Fabric workspace...")

def workspace_menu(user_id, username, cursor, conn, known_faces, known_ids, known_usernames):
    """Displays the workspace menu after successful login"""
    while True:
        print("\n=== Workspace Menu ===")
        print(f"Welcome, {username}!")
        print("1. Object Detection")
        print("2. Room Estimation")
        print("3. Logout")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            detect_and_analyze_objects(conn, cursor, user_id, username, capture_image)
        elif choice == '2':
            estimate_room(conn, cursor, user_id, username)
        elif choice == '3':
            logout(user_id)
            return False  # Return to main menu
        else:
            print("Invalid choice")
    return True

def main():
    global conn, cursor, known_faces, known_ids, known_usernames
    
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

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        print("Successfully connected to Microsoft Fabric SQL database")
        
        # Create tables
        create_tables(cursor)
        
        # Initialize known faces data
        known_faces, known_ids, known_usernames = load_known_faces(cursor)
        
        while True:
            print("\n=== Face Recognition System ===")
            print("1. Enroll User")
            print("2. Login")
            print("3. Delete User")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                username = input("Enter username: ").strip()
                email = input("Enter email: ").strip()
                enroll_user(username, email, cursor, conn, known_faces, known_ids, known_usernames)
                # Reload known faces after enrollment
                known_faces, known_ids, known_usernames = load_known_faces(cursor)
                
            elif choice == '2':
                username = input("Enter username: ").strip()
                user_id, username = login(username, cursor, conn, known_faces, known_ids, known_usernames)
                if user_id and username:
                    print(f"Login successful! Welcome {username}")
                    # Log workspace access
                    cursor.execute("""
                        INSERT INTO workspace_access_logs (
                            user_id, 
                            username, 
                            access_timestamp
                        ) VALUES (?, ?, ?)
                    """, (
                        user_id,
                        username,
                        format_timestamp()
                    ))
                    conn.commit()
                    # Open MS Fabric workspace
                    open_ms_fabric_workspace()
                    # Enter workspace menu
                    if not workspace_menu(user_id, username, cursor, conn, known_faces, known_ids, known_usernames):
                        user_id = None
                        username = None
                else:
                    print("Login failed")
                
            elif choice == '3':
                username = input("Enter username to delete: ").strip()
                delete_user(username, cursor, conn, known_faces, known_ids, known_usernames)
                # Reload known faces after deletion
                known_faces, known_ids, known_usernames = load_known_faces(cursor)
                
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("Database connection closed")

if __name__ == "__main__":
    main()