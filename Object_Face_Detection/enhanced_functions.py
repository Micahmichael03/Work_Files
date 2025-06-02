import re
import cv2
import numpy as np
import face_recognition
from pyzbar.pyzbar import decode
from PIL import Image
import os
from datetime import datetime
import qrcode
import base64
from class_config import draw_bounding_box, get_color
from common_functions import (
    capture_image,
    complete_enrollment,
    generate_user_id,
    is_face_enrolled,
    format_timestamp,
    load_known_faces,
    process_login_image,
    process_login_qr
)

def enroll_user(username, email, cursor, conn, known_faces, known_ids, known_usernames):
    """
    Enhanced enrollment function with video support and bounding boxes
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
        process_enrollment_image(image, temp_filename, username, email, cursor, conn, known_faces, known_ids, known_usernames)
    elif capture_choice == '2':
        process_enrollment_video(username, email, cursor, conn, known_faces, known_ids, known_usernames)
    else:
        print("Invalid choice")

def process_enrollment_image(image, temp_filename, username, email, cursor, conn, known_faces, known_ids, known_usernames):
    """Process enrollment from a single image"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    # Draw bounding boxes
    for (top, right, bottom, left) in face_locations:
        image = draw_bounding_box(image, (left, top, right, bottom), "Face", color=(0, 255, 0))
    
    if len(face_locations) != 1:
        print("Please ensure only one face is in the image")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return
        
    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    existing_user_id, existing_username = is_face_enrolled(face_encoding, known_faces, known_ids, known_usernames)
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

def process_enrollment_video(username, email, cursor, conn, known_faces, known_ids, known_usernames):
    """Process enrollment from video stream"""
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
                existing_user_id, existing_username = is_face_enrolled(face_encoding, known_faces, known_ids, known_usernames)
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
    """Enhanced login function with video support and bounding boxes"""
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

def process_face_login(username, use_video, cursor, conn, known_faces, known_ids, known_usernames):
    """Process face-based login"""
    if use_video:
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

def process_qr_login(username, use_video, cursor, conn):
    """Process QR code-based login"""
    if use_video:
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
        return process_login_qr(image, temp_filename, username, cursor, conn)

def delete_user(username, cursor, conn, known_faces, known_ids, known_usernames):
    """Enhanced delete user function with video verification"""
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
        cursor.execute("DELETE FROM login_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM workspace_access_logs WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        
        qr_files = [f for f in os.listdir("qr_codes") if f.startswith(f"{username}_")]
        for qr_file in qr_files:
            try:
                os.remove(os.path.join("qr_codes", qr_file))
            except:
                pass
                
        print(f"User {username} (ID: {user_id}, Email: {email}) deleted successfully")
        load_known_faces(cursor)
    except Exception as e:
        print(f"Error deleting user: {e}")

def verify_face_deletion(username, use_video, known_faces, known_ids, known_usernames):
    """Verify user identity for deletion using face recognition"""
    if use_video:
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

def verify_qr_deletion(username, use_video):
    """Verify user identity for deletion using QR code"""
    if use_video:
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