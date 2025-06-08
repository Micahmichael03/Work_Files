from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import supervision as sv
import json
 
app = Flask(__name__)

# Initialize YOLO model
try:
    model = YOLO('yolov8n.pt')
except:
    model = None

# Initialize Supervision annotators
box_annotator = sv.BoxAnnotator(
    thickness=2
)

# Global variables
camera = None
captured_frame = None
detection_results = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def process_frame(frame):
    if model is None:
        return frame, []
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLOv8 inference
    results = model(frame_rgb)
    
    # Get detections if any exist
    if len(results[0].boxes) > 0:
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        
        # Annotate frame
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)
        ]
        
        annotated_frame = box_annotator.annotate(
            scene=frame_rgb.copy(),
            detections=detections,
            labels=labels
        )
        
        return annotated_frame, detections
    
    return frame_rgb, []

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, _ = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global captured_frame, detection_results
    camera = get_camera()
    success, frame = camera.read()
    if success:
        processed_frame, detections = process_frame(frame)
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Store detection results with bounding boxes
        detection_results = []
        if len(detections) > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                detection_results.append({
                    'class': model.model.names[detections.class_id[i]],
                    'confidence': float(detections.confidence[i]),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        return jsonify({
            'success': True,
            'image': img_str,
            'detections': detection_results
        })
    return jsonify({'success': False})

@app.route('/analyze', methods=['POST'])
def analyze():
    global captured_frame, detection_results
    if detection_results is None:
        return jsonify({'success': False, 'error': 'No captured image'})
    
    # Process detections and return analysis
    return jsonify({
        'success': True,
        'detections': detection_results
    })

@app.route('/stop', methods=['POST'])
def stop():
    release_camera()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 