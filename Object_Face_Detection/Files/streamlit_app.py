# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import face_recognition
import qrcode
from datetime import datetime
import json
import base64
from io import BytesIO
import os
import tempfile
from ultralytics import YOLO
import supervision as sv

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

# Initialize YOLO model
try:
    model = YOLO('yolov8n.pt')
except:
    st.error("Failed to load YOLOv8 model. Using placeholder detection.")
    model = None

# Initialize Supervision annotators
box_annotator = sv.BoxAnnotator(
    thickness=2
)

# Initialize database connection
def init_db():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT,
            face_encoding TEXT,
            qr_code TEXT,
            created_at TIMESTAMP
        )
    ''')
    
    # Create object_detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            object_name TEXT,
            dimensions TEXT,
            cost_estimate REAL,
            qr_code TEXT,
            detected_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    return conn, cursor

# Initialize database
conn, cursor = init_db()

# Load known faces
def load_known_faces(cursor):
    cursor.execute("SELECT user_id, face_encoding, username FROM users WHERE face_encoding IS NOT NULL")
    known_faces = []
    known_ids = []
    known_usernames = []
    
    for row in cursor.fetchall():
        try:
            face_encoding = np.frombuffer(base64.b64decode(row[1]), dtype=np.float64)
            known_faces.append(face_encoding)
            known_ids.append(row[0])
            known_usernames.append(row[2])
        except:
            continue
    
    return known_faces, known_ids, known_usernames

known_faces, known_ids, known_usernames = load_known_faces(cursor)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        color: #333333;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTitle {
        color: #1E3A8A;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .stSubheader {
        color: #1E3A8A;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 15px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)

# Main navigation
st.sidebar.title("Navigation")
if not st.session_state.is_logged_in:
    page = st.sidebar.radio("Go to", ["Login", "Enroll"])
else:
    page = st.sidebar.radio("Go to", ["Home", "Object Detection", "Face Management", "Weather Data", "Logout"])

# Login page
if page == "Login":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>👤 Login</h1>", unsafe_allow_html=True)
    
    username = st.text_input("Username")
    login_method = st.radio("Login Method", ["Face Recognition", "QR Code"])
    
    if st.button("Login"):
        if login_method == "Face Recognition":
            # Process face login
            uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Detect face
                face_locations = face_recognition.face_locations(image_np)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image_np, face_locations)[0]
                    
                    # Check if face is enrolled
                    matches = face_recognition.compare_faces(known_faces, face_encoding)
                    if True in matches:
                        match_index = matches.index(True)
                        st.session_state.user_id = known_ids[match_index]
                        st.session_state.username = known_usernames[match_index]
                        st.session_state.is_logged_in = True
                        st.success("Login successful!")
                        st.experimental_rerun()
                    else:
                        st.error("Face not recognized. Please enroll first.")
                else:
                    st.error("No face detected in the image.")
        else:
            # Process QR login
            uploaded_file = st.file_uploader("Upload QR code", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Decode QR code
                try:
                    qr = qrcode.QRCode()
                    qr.add_data(image_np)
                    qr.make()
                    qr_data = qr.get_data()
                    
                    # Check if QR code matches user
                    cursor.execute("SELECT user_id, username FROM users WHERE qr_code = ?", (qr_data,))
                    result = cursor.fetchone()
                    if result:
                        st.session_state.user_id = result[0]
                        st.session_state.username = result[1]
                        st.session_state.is_logged_in = True
                        st.success("Login successful!")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid QR code.")
                except:
                    st.error("Failed to decode QR code.")

# Enrollment page
elif page == "Enroll":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📝 Enroll New User</h1>", unsafe_allow_html=True)
    
    username = st.text_input("Username")
    email = st.text_input("Email")
    enrollment_method = st.radio("Enrollment Method", ["Image Upload", "Video Capture"])
    
    if st.button("Enroll"):
        if enrollment_method == "Image Upload":
            uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Detect face
                face_locations = face_recognition.face_locations(image_np)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image_np, face_locations)[0]
                    
                    # Generate user ID
                    user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # Generate QR code
                    qr = qrcode.QRCode()
                    qr.add_data(user_id)
                    qr.make()
                    qr_img = qr.make_image()
                    
                    # Convert QR code to base64
                    buffered = BytesIO()
                    qr_img.save(buffered, format="PNG")
                    qr_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Store user data
                    cursor.execute("""
                        INSERT INTO users (user_id, username, email, face_encoding, qr_code, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        username,
                        email,
                        base64.b64encode(face_encoding.tobytes()).decode(),
                        qr_base64,
                        datetime.now()
                    ))
                    conn.commit()
                    
                    st.success("Enrollment successful! You can now login.")
                else:
                    st.error("No face detected in the image.")
        else:
            st.error("Video capture enrollment coming soon!")

# Home page
elif page == "Home":
    st.markdown(f"<h1 style='text-align: center; color: #1E3A8A;'>Welcome, {st.session_state.username}!</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; max-width: 800px; margin: 0 auto 2rem auto;'>
            <p style='font-size: 1.2rem; color: #475569;'>
                Use the navigation menu to access different features:
            </p>
            <ul style='text-align: left; margin: 1rem auto; max-width: 400px;'>
                <li>Object Detection: Detect and measure construction objects</li>
                <li>Face Management: Manage your face recognition settings</li>
                <li>Weather Data: View weather information</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Object Detection page
elif page == "Object Detection":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🔍 Object Detection</h1>", unsafe_allow_html=True)
    
    detection_method = st.radio("Detection Method", ["Image Upload", "Live Camera"])
    
    if detection_method == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Objects"):
                with st.spinner("Detecting objects..."):
                    # Convert PIL Image to numpy array
                    image_np = np.array(image)
                    
                    # Run YOLOv8 inference
                    results = model(image_np)
                    
                    # Get detections if any exist
                    if len(results[0].boxes) > 0:
                        detections = sv.Detections(
                            xyxy=results[0].boxes.xyxy.cpu().numpy(),
                            confidence=results[0].boxes.conf.cpu().numpy(),
                            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                        )
                        
                        # Annotate image with bounding boxes
                        labels = [
                            f"{model.model.names[class_id]} {confidence:0.2f}"
                            for class_id, confidence in zip(detections.class_id, detections.confidence)
                        ]
                        
                        annotated_image = box_annotator.annotate(
                            scene=image_np.copy(),
                            detections=detections,
                            labels=labels
                        )
                        
                        st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                        
                        # Store detection results
                        for i in range(len(detections)):
                            object_name = model.model.names[detections.class_id[i]]
                            confidence = detections.confidence[i]
                            
                            cursor.execute("""
                                INSERT INTO object_detections (user_id, object_name, detected_at)
                                VALUES (?, ?, ?)
                            """, (
                                st.session_state.user_id,
                                object_name,
                                datetime.now()
                            ))
                        conn.commit()
                    else:
                        st.warning("No objects detected in the image.")
    
    else:  # Live Camera
        if st.button("Start Camera"):
            st.warning("Live camera detection coming soon!")

# Face Management page
elif page == "Face Management":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>👤 Face Management</h1>", unsafe_allow_html=True)
    
    face_action = st.radio("Select Action", ["Update Face", "Delete Face"])
    
    if face_action == "Update Face":
        if st.button("Update Face Recognition"):
            st.warning("Face update coming soon!")
    
    else:  # Delete Face
        if st.button("Delete Face Recognition"):
            cursor.execute("DELETE FROM users WHERE user_id = ?", (st.session_state.user_id,))
            conn.commit()
            
            st.session_state.is_logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.success("Face recognition deleted successfully!")
            st.experimental_rerun()

# Weather Data page
elif page == "Weather Data":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🌤️ Weather Data</h1>", unsafe_allow_html=True)
    st.info("Weather data functionality coming soon!")

# Logout
elif page == "Logout":
    if st.button("Logout"):
        st.session_state.is_logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.success("Logged out successfully!")
        st.experimental_rerun()

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <p class='footer-text'>Powered by YOLOv8 object detection | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
