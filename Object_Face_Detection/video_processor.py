import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
from pyzbar.pyzbar import decode
from PIL import Image
from class_config import draw_bounding_box, get_color

class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize video processor with YOLO model"""
        self.model = YOLO(model_path)
        self.known_faces = []
        self.known_ids = []
        self.known_usernames = []
        
    def load_known_faces(self, faces, ids, usernames):
        """Load known faces for recognition"""
        self.known_faces = faces
        self.known_ids = ids
        self.known_usernames = usernames
        
    def process_frame(self, frame, mode='object_detection'):
        """
        Process a single frame based on the selected mode
        Modes: 'object_detection', 'face_recognition', 'qr_code'
        """
        if mode == 'object_detection':
            return self._process_object_detection(frame)
        elif mode == 'face_recognition':
            return self._process_face_recognition(frame)
        elif mode == 'qr_code':
            return self._process_qr_code(frame)
        return frame
        
    def _process_object_detection(self, frame):
        """Process frame for object detection"""
        results = self.model(frame, conf=0.25)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.25:
                    object_name = result.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    frame = draw_bounding_box(frame, (x1, y1, x2, y2), object_name, conf)
        
        return frame
        
    def _process_face_recognition(self, frame):
        """Process frame for face recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if self.known_faces:
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_usernames[first_match_index]
                    frame = draw_bounding_box(frame, (left, top, right, bottom), f"Face: {name}", color=(0, 255, 0))
            else:
                frame = draw_bounding_box(frame, (left, top, right, bottom), "Unknown Face", color=(0, 255, 0))
        
        return frame
        
    def _process_qr_code(self, frame):
        """Process frame for QR code detection"""
        decoded_objects = decode(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        for obj in decoded_objects:
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull
            n = len(points)
            for j in range(n):
                cv2.line(frame, tuple(points[j]), tuple(points[(j+1) % n]), (0, 255, 0), 3)
            
            x, y, w, h = obj.rect
            frame = draw_bounding_box(frame, (x, y, x+w, y+h), "QR Code", color=(255, 0, 0))
            
        return frame
        
    def process_video(self, source=0, mode='object_detection', output_path=None):
        """
        Process video from source (webcam or file)
        Args:
            source: Video source (0 for webcam, or path to video file)
            mode: Processing mode ('object_detection', 'face_recognition', 'qr_code')
            output_path: Path to save processed video (optional)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error opening video source {source}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Processing video in {mode} mode. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame, mode)
            
            # Display frame
            cv2.imshow('Video Processing', processed_frame)
            
            # Write frame if output path is provided
            if writer:
                writer.write(processed_frame)
                
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows() 