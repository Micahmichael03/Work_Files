# Class names and their corresponding colors for visualization
import cv2


CLASS_COLORS = {
    # Face Recognition Colors
    'face': (0, 255, 0),  # Green for faces
    'qr_code': (255, 0, 0),  # Blue for QR codes
    
    # YOLO Object Detection Colors
    'person': (255, 0, 0),  # Red
    'chair': (0, 255, 0),   # Green
    'table': (0, 0, 255),   # Blue
    'laptop': (255, 255, 0), # Cyan
    'mouse': (255, 0, 255), # Magenta
    'keyboard': (0, 255, 255), # Yellow
    'book': (128, 0, 0),    # Dark Red
    'bottle': (0, 128, 0),  # Dark Green
    'cup': (0, 0, 128),     # Dark Blue
    'phone': (128, 128, 0), # Dark Yellow
    'tv': (128, 0, 128),    # Dark Magenta
    'couch': (0, 128, 128), # Dark Cyan
    'bed': (64, 64, 64),    # Dark Gray
    'clock': (192, 192, 192), # Light Gray
    'vase': (255, 128, 0),  # Orange
    'potted plant': (128, 255, 0), # Light Green
    'dining table': (0, 128, 255), # Light Blue
    'toilet': (255, 0, 128), # Pink
    'sink': (128, 0, 255),  # Purple
    'refrigerator': (0, 255, 128), # Mint
    'microwave': (255, 128, 128), # Light Red
    'oven': (128, 255, 128), # Light Green
    'toaster': (128, 128, 255), # Light Blue
}

# Default color for unknown classes
DEFAULT_COLOR = (255, 255, 255)  # White

def get_color(class_name):
    """Returns the color for a given class name"""
    return CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)

def draw_bounding_box(image, box, class_name, confidence=None, color=None):
    """
    Draws a bounding box with class name and confidence on the image
    Args:
        image: The image to draw on
        box: (x1, y1, x2, y2) coordinates
        class_name: Name of the detected class
        confidence: Detection confidence (optional)
        color: BGR color tuple (optional)
    """
    if color is None:
        color = get_color(class_name)
    
    x1, y1, x2, y2 = map(int, box)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label
    label = class_name
    if confidence is not None:
        label += f' {confidence:.2f}'
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # Draw label background
    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image 