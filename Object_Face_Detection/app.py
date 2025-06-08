import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
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

def run_live_camera_capture():
    """Run live camera with capture functionality"""
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>📷 Live Camera Capture</h2>", unsafe_allow_html=True)
    
    # Initialize session state for camera
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None

    # Camera controls
    st.markdown("""
        <style>
        /* Camera Control Buttons */
        div[data-testid="stButton"] button {
            background-color: #1E3A8A;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div[data-testid="stButton"] button:hover {
            background-color: #1E40AF;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        div[data-testid="stButton"] button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div[data-testid="stButton"] button.capture {
            background-color: #059669;
            padding: 0.75rem 1.5rem;
            margin-top: 1rem;
        }
        div[data-testid="stButton"] button.capture:hover {
            background-color: #047857;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    start_button = col1.button("🎥 Start Camera")
    stop_button = col2.button("⏹️ Stop Camera")
    capture_button = st.button("📸 Capture Image", key="capture")

    if start_button:
        st.session_state.camera_on = True
        st.session_state.captured_image = None

    if stop_button:
        st.session_state.camera_on = False
        st.session_state.captured_image = None

    # Camera feed and capture
    if st.session_state.camera_on:
        camera_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Display the live camera feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Handle capture
            if capture_button:
                st.session_state.captured_image = frame_rgb
                cap.release()
                st.session_state.camera_on = False
                st.rerun()
                break

    # Show captured image and analysis
    if st.session_state.captured_image is not None:
        st.markdown("<div class='detection-results'>", unsafe_allow_html=True)
        st.image(st.session_state.captured_image, caption="Captured Image", use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze Captured Image"):
            with st.spinner("Processing image..."):
                # Process image
                annotated_image, detections = process_frame(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_RGB2BGR))
                
                # Show detection results
                st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                
                # Get detected items
                detected_items = {}
                if len(detections) > 0:
                    for i in range(len(detections)):
                        class_id = detections.class_id[i]
                        class_name = model.model.names[class_id]
                        detected_items[class_name] = detected_items.get(class_name, 0) + 1
                    
                    if detected_items:
                        st.markdown("""
                            <div class='success-box'>
                                <h3 style='margin: 0; color: #1E3A8A;'>✅ Objects Detected</h3>
                                <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>{}</p>
                            </div>
                        """.format(', '.join([f'<strong style="color: #1E3A8A;">{k}</strong> ({v})' for k, v in detected_items.items()])), unsafe_allow_html=True)
                        
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
                                    <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>No matching construction items found in our database.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class='warning-box'>
                                <h3 style='margin: 0; color: #2E4057;'>⚠️ No Objects Detected</h3>
                                <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>No construction items detected in the image.</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='warning-box'>
                            <h3 style='margin: 0; color: #2E4057;'>⚠️ No Objects Detected</h3>
                            <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>No construction items detected in the image.</p>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
    <style>
    /* Camera Control Buttons */
    div[data-testid="stButton"] button {
        background-color: #1E3A8A;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] button:hover {
        background-color: #1E40AF;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] button.capture {
        background-color: #059669;
        padding: 0.75rem 1.5rem;
        margin-top: 1rem;
    }
    div[data-testid="stButton"] button.capture:hover {
        background-color: #047857;
    }
    
    /* Object Detection Display Styles */
    .success-box {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box p {
        color: #1E293B;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .success-box strong {
        color: #1E3A8A;
    }
    
    .warning-box {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box p {
        color: #1E293B;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .item-box {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .item-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    
    .item-details {
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    .buy-button {
        background-color: #059669;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .buy-button:hover {
        background-color: #047857;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .cost-box {
        background-color: #F8FAFC;
        border: 2px solid #1E3A8A;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏗️ Construction Object Detection & Cost Estimator</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; max-width: 800px; margin: 0 auto 2rem auto;'>
        <p style='font-size: 1.2rem; color: #475569;'>
            Use live camera or upload images to detect construction objects and get cost estimates.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Live camera capture option
run_live_camera_capture()

st.markdown("---")

# Original upload option (unchanged)
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

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("<div class='detection-results'>", unsafe_allow_html=True)
    
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process button
    if st.button("🔍 Analyze Uploaded Image"):
        # Convert to OpenCV format
        image_np = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Processing image..."):
            # Process image
            annotated_image, detections = process_frame(image_cv)
            
            # Show detection results
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)
            
            # Get detected items
            detected_items = {}
            if len(detections) > 0:
                for i in range(len(detections)):
                    class_id = detections.class_id[i]
                    class_name = model.model.names[class_id]
                    detected_items[class_name] = detected_items.get(class_name, 0) + 1
                
                if detected_items:
                    st.markdown("""
                        <div class='success-box'>
                            <h3 style='margin: 0; color: #1E3A8A;'>✅ Objects Detected</h3>
                            <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>{}</p>
                        </div>
                    """.format(', '.join([f'<strong style="color: #1E3A8A;">{k}</strong> ({v})' for k, v in detected_items.items()])), unsafe_allow_html=True)
                    
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
                                <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>No matching construction items found in our database.</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='warning-box'>
                            <h3 style='margin: 0; color: #2E4057;'>⚠️ No Objects Detected</h3>
                            <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>No construction items detected in the image.</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='warning-box'>
                        <h3 style='margin: 0; color: #2E4057;'>⚠️ No Objects Detected</h3>
                        <p style='margin: 0.5rem 0 0 0; color: #1E293B;'>No construction items detected in the image.</p>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <p class='footer-text'>Powered by YOLOv8 object detection | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)