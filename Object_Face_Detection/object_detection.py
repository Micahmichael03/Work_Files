import cv2
import numpy as np
from ultralytics import YOLO
import json
from io import BytesIO
import base64
import qrcode
import webbrowser
import os
import uuid
from datetime import datetime
import pyodbc 

from common_functions import detect_a4_paper, get_cost_estimate

def detect_and_measure_object(image):
    """
    Detects objects in the image, measures their dimensions, and estimates costs
    Returns object information including dimensions, type, and cost estimates
    """
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    
    # Convert image to RGB for YOLO
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform object detection
    results = model(rgb_image)
    
    detected_objects = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get object class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf > 0.5:  # Confidence threshold
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
    """
    Stores object detection results in the database
    """
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
            datetime.now().strftime("%m/%d/%Y %I:%M %p")
        ))
        
        conn.commit()
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