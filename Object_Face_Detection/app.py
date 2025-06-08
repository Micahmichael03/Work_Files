import streamlit as st
import random
import time
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Initialize YOLO model
try:
    model = YOLO('yolov8n.pt')  # You can replace with your custom trained model
except:
    st.error("Failed to load YOLOv8 model. Using placeholder detection.")
    model = None

# Initialize Supervision annotators
box_annotator = sv.BoxAnnotator(
    thickness=2
)

# Enhanced construction object database with 80+ items
construction_items = [
    # Structural Materials
    {"object": "Drywall", "unit_cost": 15, "supplier": "https://homehardware.com/drywall"},
    {"object": "Plywood", "unit_cost": 50, "supplier": "https://homedepot.com/plywood"},
    {"object": "2x4 Lumber", "unit_cost": 8, "supplier": "https://lowes.com/lumber"},
    {"object": "Concrete Block", "unit_cost": 3, "supplier": "https://menards.com/concrete-block"},
    {"object": "Rebar", "unit_cost": 0.75, "supplier": "https://homedepot.com/rebar"},
    
    # Fixtures
    {"object": "Ceiling Fan", "unit_cost": 120, "supplier": "https://homedepot.com/ceiling-fan"},
    {"object": "Sink", "unit_cost": 200, "supplier": "https://lowes.com/sink"},
    {"object": "Toilet", "unit_cost": 250, "supplier": "https://homedepot.com/toilet"},
    {"object": "Shower Head", "unit_cost": 85, "supplier": "https://lowes.com/shower-head"},
    {"object": "Faucet", "unit_cost": 90, "supplier": "https://homedepot.com/faucet"},
    
    # Electrical
    {"object": "Light Switch", "unit_cost": 5, "supplier": "https://lowes.com/light-switch"},
    {"object": "Outlet", "unit_cost": 7, "supplier": "https://homedepot.com/outlet"},
    {"object": "Circuit Breaker", "unit_cost": 25, "supplier": "https://menards.com/circuit-breaker"},
    {"object": "Wire (per ft)", "unit_cost": 0.50, "supplier": "https://homedepot.com/wire"},
    {"object": "Light Fixture", "unit_cost": 45, "supplier": "https://lowes.com/light-fixture"},
    
    # Plumbing
    {"object": "PVC Pipe", "unit_cost": 2.50, "supplier": "https://homedepot.com/pvc-pipe"},
    {"object": "Copper Pipe", "unit_cost": 8, "supplier": "https://lowes.com/copper-pipe"},
    {"object": "Pipe Fitting", "unit_cost": 3, "supplier": "https://menards.com/pipe-fitting"},
    {"object": "Water Heater", "unit_cost": 600, "supplier": "https://homedepot.com/water-heater"},
    {"object": "Sump Pump", "unit_cost": 150, "supplier": "https://lowes.com/sump-pump"},
    
    # Tools
    {"object": "Hammer", "unit_cost": 25, "supplier": "https://homedepot.com/hammer"},
    {"object": "Drill", "unit_cost": 80, "supplier": "https://lowes.com/drill"},
    {"object": "Circular Saw", "unit_cost": 120, "supplier": "https://menards.com/circular-saw"},
    {"object": "Ladder", "unit_cost": 150, "supplier": "https://homedepot.com/ladder"},
    {"object": "Toolbox", "unit_cost": 45, "supplier": "https://lowes.com/toolbox"},
    
    # Safety Equipment
    {"object": "Hard Hat", "unit_cost": 20, "supplier": "https://homedepot.com/hard-hat"},
    {"object": "Safety Glasses", "unit_cost": 10, "supplier": "https://lowes.com/safety-glasses"},
    {"object": "Work Gloves", "unit_cost": 15, "supplier": "https://menards.com/work-gloves"},
    {"object": "Safety Vest", "unit_cost": 12, "supplier": "https://homedepot.com/safety-vest"},
    {"object": "Ear Protection", "unit_cost": 18, "supplier": "https://lowes.com/ear-protection"},
    
    # Finishing Materials
    {"object": "Paint (gallon)", "unit_cost": 35, "supplier": "https://homedepot.com/paint"},
    {"object": "Wallpaper", "unit_cost": 25, "supplier": "https://lowes.com/wallpaper"},
    {"object": "Baseboard", "unit_cost": 2.50, "supplier": "https://menards.com/baseboard"},
    {"object": "Crown Molding", "unit_cost": 3.50, "supplier": "https://homedepot.com/crown-molding"},
    {"object": "Tile (sq ft)", "unit_cost": 4, "supplier": "https://lowes.com/tile"},
    
    # Doors and Windows
    {"object": "Interior Door", "unit_cost": 150, "supplier": "https://homedepot.com/interior-door"},
    {"object": "Exterior Door", "unit_cost": 300, "supplier": "https://lowes.com/exterior-door"},
    {"object": "Window", "unit_cost": 250, "supplier": "https://menards.com/window"},
    {"object": "Sliding Glass Door", "unit_cost": 500, "supplier": "https://homedepot.com/sliding-door"},
    {"object": "Storm Door", "unit_cost": 200, "supplier": "https://lowes.com/storm-door"},
    
    # HVAC
    {"object": "Furnace", "unit_cost": 1200, "supplier": "https://homedepot.com/furnace"},
    {"object": "Air Conditioner", "unit_cost": 800, "supplier": "https://lowes.com/air-conditioner"},
    {"object": "Thermostat", "unit_cost": 100, "supplier": "https://menards.com/thermostat"},
    {"object": "Air Filter", "unit_cost": 15, "supplier": "https://homedepot.com/air-filter"},
    {"object": "Ductwork (per ft)", "unit_cost": 3, "supplier": "https://lowes.com/ductwork"},
    
    # Roofing
    {"object": "Asphalt Shingles", "unit_cost": 1.50, "supplier": "https://homedepot.com/shingles"},
    {"object": "Roofing Nails", "unit_cost": 0.05, "supplier": "https://lowes.com/roofing-nails"},
    {"object": "Flashing", "unit_cost": 8, "supplier": "https://menards.com/flashing"},
    {"object": "Gutter", "unit_cost": 4, "supplier": "https://homedepot.com/gutter"},
    {"object": "Downspout", "unit_cost": 5, "supplier": "https://lowes.com/downspout"},
    
    # Landscaping
    {"object": "Pavers", "unit_cost": 0.75, "supplier": "https://homedepot.com/pavers"},
    {"object": "Mulch", "unit_cost": 3, "supplier": "https://lowes.com/mulch"},
    {"object": "Topsoil", "unit_cost": 2, "supplier": "https://menards.com/topsoil"},
    {"object": "Landscape Timber", "unit_cost": 6, "supplier": "https://homedepot.com/landscape-timber"},
    {"object": "Decorative Stone", "unit_cost": 5, "supplier": "https://lowes.com/decorative-stone"},
    
    # Hardware
    {"object": "Nails (per lb)", "unit_cost": 4, "supplier": "https://homedepot.com/nails"},
    {"object": "Screws (per lb)", "unit_cost": 5, "supplier": "https://lowes.com/screws"},
    {"object": "Bolts", "unit_cost": 0.50, "supplier": "https://menards.com/bolts"},
    {"object": "Hinges", "unit_cost": 3, "supplier": "https://homedepot.com/hinges"},
    {"object": "Drawer Pull", "unit_cost": 2.50, "supplier": "https://lowes.com/drawer-pull"},
    
    # Concrete and Masonry
    {"object": "Concrete Mix", "unit_cost": 6, "supplier": "https://homedepot.com/concrete-mix"},
    {"object": "Mortar", "unit_cost": 5, "supplier": "https://lowes.com/mortar"},
    {"object": "Bricks", "unit_cost": 0.75, "supplier": "https://menards.com/bricks"},
    {"object": "Sand", "unit_cost": 4, "supplier": "https://homedepot.com/sand"},
    {"object": "Gravel", "unit_cost": 3, "supplier": "https://lowes.com/gravel"},
    
    # Insulation
    {"object": "Fiberglass Insulation", "unit_cost": 0.50, "supplier": "https://homedepot.com/insulation"},
    {"object": "Foam Board", "unit_cost": 1.25, "supplier": "https://lowes.com/foam-board"},
    {"object": "Spray Foam", "unit_cost": 8, "supplier": "https://menards.com/spray-foam"},
    {"object": "Vapor Barrier", "unit_cost": 0.30, "supplier": "https://homedepot.com/vapor-barrier"},
    {"object": "Weather Stripping", "unit_cost": 0.25, "supplier": "https://lowes.com/weather-stripping"},
    
    # Cabinetry
    {"object": "Kitchen Cabinet", "unit_cost": 200, "supplier": "https://homedepot.com/kitchen-cabinet"},
    {"object": "Bathroom Vanity", "unit_cost": 300, "supplier": "https://lowes.com/bathroom-vanity"},
    {"object": "Drawer Slides", "unit_cost": 15, "supplier": "https://menards.com/drawer-slides"},
    {"object": "Cabinet Knob", "unit_cost": 2, "supplier": "https://homedepot.com/cabinet-knob"},
    {"object": "Pantry Shelf", "unit_cost": 25, "supplier": "https://lowes.com/pantry-shelf"},
    
    # Flooring
    {"object": "Hardwood Flooring", "unit_cost": 8, "supplier": "https://homedepot.com/hardwood-flooring"},
    {"object": "Laminate Flooring", "unit_cost": 3, "supplier": "https://lowes.com/laminate-flooring"},
    {"object": "Vinyl Plank", "unit_cost": 2.50, "supplier": "https://menards.com/vinyl-plank"},
    {"object": "Carpet (sq yd)", "unit_cost": 15, "supplier": "https://homedepot.com/carpet"},
    {"object": "Underlayment", "unit_cost": 0.75, "supplier": "https://lowes.com/underlayment"},
    
    # Additional items to reach 80+
    {"object": "Drywall Screws", "unit_cost": 0.10, "supplier": "https://homedepot.com/drywall-screws"},
    {"object": "Joint Compound", "unit_cost": 12, "supplier": "https://lowes.com/joint-compound"},
    {"object": "Tape Measure", "unit_cost": 15, "supplier": "https://menards.com/tape-measure"},
    {"object": "Level", "unit_cost": 20, "supplier": "https://homedepot.com/level"},
    {"object": "Utility Knife", "unit_cost": 10, "supplier": "https://lowes.com/utility-knife"},
]

def detect_objects(image):
    """Detect objects in image using YOLOv8 and return annotated image and detections"""
    if model is None:
        return image, []
    
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
        
        # Filter for construction items we have in our database
        construction_item_names = [item["object"] for item in construction_items]
        filtered_detections = []
        
        for i in range(len(detections)):
            class_name = model.model.names[detections.class_id[i]]
            if class_name in construction_item_names:
                filtered_detections.append((
                    detections.xyxy[i],
                    detections.confidence[i],
                    detections.class_id[i]
                ))
        
        if filtered_detections:
            # Annotate image with bounding boxes
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id in filtered_detections
            ]
            
            annotated_image = box_annotator.annotate(
                scene=image_np.copy(),
                detections=sv.Detections(
                    xyxy=np.array([d[0] for d in filtered_detections]),
                    confidence=np.array([d[1] for d in filtered_detections]),
                    class_id=np.array([d[2] for d in filtered_detections])
                ),
                labels=labels
            )
            
            # Get detected items with counts
            detected_items = {}
            for _, _, class_id in filtered_detections:
                class_name = model.model.names[class_id]
                detected_items[class_name] = detected_items.get(class_name, 0) + 1
            
            return Image.fromarray(annotated_image), detected_items
    
    # If no detections or no filtered detections, return original image
    return image, {}

def run_live_camera_detection():
    """Run live camera detection with YOLOv8"""
    st.write("Live Camera Detection - Press 'Stop' to end")
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    stop_button = col1.button("Stop Camera")
    capture_button = col2.button("Capture & Analyze")
    
    # Create placeholder for analysis results
    analysis_placeholder = st.empty()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open camera")
        return
    
    frame_window = st.image([])
    captured_frame = None
    
    while not stop_button:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if model:
            # Run detection
            results = model(frame)
            
            # Only process if we have detections
            if len(results[0].boxes) > 0:
                detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                
                # Filter for construction items
                construction_item_names = [item["object"] for item in construction_items]
                filtered_detections = []
                
                for i in range(len(detections)):
                    class_name = model.model.names[detections.class_id[i]]
                    if class_name in construction_item_names:
                        filtered_detections.append((
                            detections.xyxy[i],
                            detections.confidence[i],
                            detections.class_id[i]
                        ))
                
                if filtered_detections:
                    # Annotate frame
                    labels = [
                        f"{model.model.names[class_id]} {confidence:0.2f}"
                        for _, confidence, class_id in filtered_detections
                    ]
                    
                    annotated_frame = box_annotator.annotate(
                        scene=frame.copy(),
                        detections=sv.Detections(
                            xyxy=np.array([d[0] for d in filtered_detections]),
                            confidence=np.array([d[1] for d in filtered_detections]),
                            class_id=np.array([d[2] for d in filtered_detections])
                        ),
                        labels=labels
                    )
                    frame_window.image(annotated_frame)
                    captured_frame = annotated_frame
                else:
                    frame_window.image(frame)
                    captured_frame = frame
            else:
                frame_window.image(frame)
                captured_frame = frame
        else:
            frame_window.image(frame)
            captured_frame = frame
        
        # Check if capture button was pressed
        if capture_button and captured_frame is not None:
            # Convert captured frame to PIL Image
            captured_image = Image.fromarray(captured_frame)
            
            # Create a temporary file to save the captured image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                captured_image.save(tmp_file.name)
                
                # Display the captured image and analysis in the placeholder
                with analysis_placeholder.container():
                    st.markdown("<h3 style='text-align: center; color: #1E3A8A;'>📸 Captured Image Analysis</h3>", unsafe_allow_html=True)
                    st.image(captured_image, caption="Captured Image", use_column_width=True)
                    
                    # Analyze the captured image
                    with st.spinner("🔍 Analyzing captured image..."):
                        # Get detected items
                        detected_items = {}
                        for _, confidence, class_id in filtered_detections:
                            class_name = model.model.names[class_id]
                            detected_items[class_name] = detected_items.get(class_name, 0) + 1
                        
                        if detected_items:
                            st.markdown("""
                                <div class='success-box'>
                                    <h3 style='margin: 0; color: #1E3A8A;'>✅ Objects Detected</h3>
                                    <p style='margin: 0.5rem 0 0 0;'>{}</p>
                                </div>
                            """.format(', '.join([f'<strong>{k}</strong> ({v})' for k, v in detected_items.items()])), unsafe_allow_html=True)
                            
                            # Find matching items in our database
                            estimates = []
                            total_cost = 0
                            
                            for item_name, quantity in detected_items.items():
                                # Find the item in our database (case insensitive)
                                matching_items = [item for item in construction_items 
                                                if item["object"].lower() == item_name.lower()]
                                
                                if matching_items:
                                    item = matching_items[0]
                                    cost = item['unit_cost'] * quantity
                                    total_cost += cost
                                    estimates.append({
                                        "object": item['object'], 
                                        "qty": quantity, 
                                        "cost": cost, 
                                        "link": item['supplier']
                                    })
                            
                            if estimates:
                                st.markdown("<h3 style='text-align: center; margin-top: 2rem; color: #1E3A8A;'>📊 Cost Estimation Results</h3>", unsafe_allow_html=True)
                                
                                for e in estimates:
                                    st.markdown(f"""
                                        <div class='item-box'>
                                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                                <div>
                                                    <span class='item-title'>{e['object']}</span><br>
                                                    <span class='item-details'>Quantity: {e['qty']} | Cost: ${e['cost']}</span>
                                                </div>
                                                <a href='{e['link']}' target='_blank' style='text-decoration: none;'>
                                                    <button class='buy-button'>Buy Now</button>
                                                </a>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class='cost-box'>
                                        <h2 style='margin: 0; color: #1E3A8A;'>💰 Total Estimated Cost</h2>
                                        <h1 style='margin: 0.5rem 0; color: #1E40AF;'>${total_cost}</h1>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class='warning-box'>
                                    <h3 style='margin: 0; color: #2E4057;'>⚠️ No Objects Detected</h3>
                                    <p style='margin: 0.5rem 0 0 0;'>No construction items detected in the captured image.</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Clean up the temporary file
                os.unlink(tmp_file.name)
        
        # Check if stop button was pressed
        if stop_button:
            break
    
    cap.release()

# Add custom CSS
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
    .success-box {
        background-color: #ECFDF5;
        padding: 1.5rem;
        border-radius: 20px;
        border-left: 5px solid #059669;
        margin: 1.5rem auto;
        color: #065F46;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.1);
        max-width: 800px;
        transition: all 0.3s ease;
    }
    .success-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.15);
    }
    .warning-box {
        background-color: #FFFBEB;
        padding: 1.5rem;
        border-radius: 20px;
        border-left: 5px solid #D97706;
        margin: 1.5rem auto;
        color: #92400E;
        box-shadow: 0 4px 15px rgba(217, 119, 6, 0.1);
        max-width: 800px;
        transition: all 0.3s ease;
    }
    .warning-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(217, 119, 6, 0.15);
    }
    .cost-box {
        background-color: #EFF6FF;
        padding: 2rem;
        border-radius: 25px;
        margin: 2rem auto;
        text-align: center;
        color: #1E40AF;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.1);
        max-width: 600px;
        transition: all 0.3s ease;
    }
    .cost-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(30, 64, 175, 0.15);
    }
    .item-box {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem auto;
        color: #1E293B;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        max-width: 800px;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    .item-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
    .camera-box {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem auto;
        text-align: center;
        color: #1E293B;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        max-width: 800px;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    .camera-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
    }
    .upload-box {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem auto;
        text-align: center;
        color: #1E293B;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        max-width: 800px;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
    }
    .footer-text {
        color: #64748B;
        text-align: center;
        margin: 2rem auto;
        max-width: 800px;
    }
    .item-title {
        color: #1E3A8A;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .item-details {
        color: #475569;
        font-size: 0.95rem;
    }
    .buy-button {
        background-color: #1E3A8A;
        color: #FFFFFF;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        cursor: pointer;
        text-decoration: none;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(30, 58, 138, 0.1);
        transition: all 0.3s ease;
    }
    .buy-button:hover {
        background-color: #1E40AF;
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(30, 58, 138, 0.2);
    }
    .detection-results {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    .upload-section {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    .camera-section {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏗️ Construction Object Detection & Cost Estimator</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; max-width: 800px; margin: 0 auto 2rem auto;'>
        <p style='font-size: 1.2rem; color: #475569;'>
            Upload an image or use live camera to detect construction objects and get instant cost estimates.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Live camera option
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>📷 Live Camera Detection</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class='camera-section'>
        <div class='camera-box'>
            <p style='text-align: center; margin-bottom: 1rem; font-size: 1.1rem;'>
                Click the button below to start real-time object detection using your camera.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

if st.button("🎥 Start Live Camera", key="live_camera"):
    run_live_camera_detection()

st.markdown("---")

# Image upload option
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>📤 Upload Image</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class='upload-section'>
        <div class='upload-box'>
            <p style='text-align: center; margin-bottom: 1rem; font-size: 1.1rem;'>
                Upload a construction site image to detect objects and get cost estimates.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

camera_input = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if camera_input:
    st.markdown("<div class='detection-results'>", unsafe_allow_html=True)
    image = Image.open(camera_input)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting objects and estimating costs..."):
        # Run object detection
        annotated_image, detected_items = detect_objects(image)
        
        # Show detection results
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)
        
        if detected_items:
            st.markdown("""
                <div class='success-box'>
                    <h3 style='margin: 0; color: #1E3A8A;'>✅ Objects Detected</h3>
                    <p style='margin: 0.5rem 0 0 0;'>{}</p>
                </div>
            """.format(', '.join([f'<strong>{k}</strong> ({v})' for k, v in detected_items.items()])), unsafe_allow_html=True)
            
            # Find matching items in our database
            estimates = []
            total_cost = 0
            
            for item_name, quantity in detected_items.items():
                # Find the item in our database (case insensitive)
                matching_items = [item for item in construction_items 
                                if item["object"].lower() == item_name.lower()]
                
                if matching_items:
                    item = matching_items[0]
                    cost = item['unit_cost'] * quantity
                    total_cost += cost
                    estimates.append({
                        "object": item['object'], 
                        "qty": quantity, 
                        "cost": cost, 
                        "link": item['supplier']
                    })
            
            if estimates:
                st.markdown("<h3 style='text-align: center; margin-top: 2rem; color: #1E3A8A;'>📊 Cost Estimation Results</h3>", unsafe_allow_html=True)
                
                for e in estimates:
                    st.markdown(f"""
                        <div class='item-box'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <span class='item-title'>{e['object']}</span><br>
                                    <span class='item-details'>Quantity: {e['qty']} | Cost: ${e['cost']}</span>
                                </div>
                                <a href='{e['link']}' target='_blank' style='text-decoration: none;'>
                                    <button class='buy-button'>Buy Now</button>
                                </a>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class='cost-box'>
                        <h2 style='margin: 0; color: #1E3A8A;'>💰 Total Estimated Cost</h2>
                        <h1 style='margin: 0.5rem 0; color: #1E40AF;'>${total_cost}</h1>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='warning-box'>
                        <h3 style='margin: 0; color: #2E4057;'>⚠️ No Matching Items</h3>
                        <p style='margin: 0.5rem 0 0 0;'>No matching construction items found in our database.</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='warning-box'>
                    <h3 style='margin: 0; color: #2E4057;'>⚠️ No Objects Detected</h3>
                    <p style='margin: 0.5rem 0 0 0;'>No construction items detected in the image.</p>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <p class='footer-text'>Powered by YOLOv8 object detection | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)