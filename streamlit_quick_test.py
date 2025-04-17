import streamlit as st
import cv2
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
import time

# Import your existing modules
from object_detection import load_model as load_detection_model, detect_objects
from semantic_segmentaiton import load_model as load_segmentation_model, predict_segmentation

def main():
    st.title("Real-time Object Detection and Segmentation")
    
    # Initialize session state for models if not already initialized
    if 'detection_model' not in st.session_state:
        with st.spinner("Loading detection model..."):
            st.session_state.detection_model = load_detection_model()
        st.success("Detection model loaded!")
    
    if 'segmentation_model' not in st.session_state:
        with st.spinner("Loading segmentation model..."):
            st.session_state.segmentation_model, st.session_state.feature_extractor = load_segmentation_model()
        st.success("Segmentation model loaded!")
    
    # Camera selection options
    camera_options = {
        "Built-in Camera": 0,
        "External Camera": 1,
        "Mobile Phone Camera (requires IP Webcam app)": "http://YOUR_PHONE_IP:8080/video"
    }
    
    camera_choice = st.selectbox(
        "Choose Camera Source", 
        list(camera_options.keys()),
        index=0
    )
    
    # Instructions for phone camera
    if "Mobile" in camera_choice:
        st.info("""
        To use your phone as a camera:
        1. Install the 'IP Webcam' app from Play Store (Android) or similar app for iOS
        2. Open the app and click 'Start server'
        3. Replace 'YOUR_PHONE_IP' in the code with your phone's IP address shown in the app
        4. Make sure your phone and computer are on the same network
        """)
    
    # Detection settings
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Visualization settings
    st.sidebar.header("Visualization Settings")
    show_segmentation = st.sidebar.checkbox("Show Segmentation", True)
    segmentation_opacity = st.sidebar.slider("Segmentation Opacity", 0.0, 1.0, 0.3, 0.05)
    
    # Camera stream capture
    stframe = st.empty()
    stop_button = st.button("Stop")
    
    camera_source = camera_options[camera_choice]
    
    # Start camera feed
    try:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_source}")
            return
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Object detection
            results = detect_objects(frame, st.session_state.detection_model, conf_threshold=confidence_threshold)
            
            if show_segmentation:
                # Save frame to temporary file for segmentation
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, frame)
                
                # Perform segmentation
                segmentation_map = predict_segmentation(
                    temp_file_path, 
                    st.session_state.segmentation_model, 
                    st.session_state.feature_extractor
                )
                
                # Resize segmentation map to match frame dimensions
                segmentation_map_resized = F.interpolate(
                    segmentation_map.unsqueeze(0).unsqueeze(0).float(),
                    size=(frame_height, frame_width),
                    mode='nearest'
                ).squeeze().long()
                
                # Create colored segmentation visualization
                segmentation_color = np.zeros_like(frame)
                num_classes = segmentation_map_resized.max().item() + 1
                
                # Generate colors for each class
                colors = {}
                for i in range(num_classes):
                    colors[i] = np.array([np.random.randint(0, 255) for _ in range(3)])
                
                # Apply colors to mask
                segmentation_np = segmentation_map_resized.cpu().numpy()
                for i in range(num_classes):
                    mask = (segmentation_np == i)
                    segmentation_color[mask] = colors[i]
                
                # Blend original frame with segmentation
                blended_frame = cv2.addWeighted(
                    frame, 1.0 - segmentation_opacity, 
                    segmentation_color, segmentation_opacity, 0
                )
                
                # Clean up temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            else:
                blended_frame = frame.copy()
            
            # Draw object detection results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(blended_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(blended_frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame in Streamlit
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # Small delay to prevent high CPU usage
            time.sleep(0.05)
            
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        st.write("Camera released")

if __name__ == "__main__":
    main()