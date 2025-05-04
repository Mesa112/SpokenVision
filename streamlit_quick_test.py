import streamlit as st
import cv2
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
import time
from datetime import datetime
import subprocess
import platform

# Import your existing modules
from object_detection import load_model as load_detection_model, detect_objects
from semantic_segmentation import load_model as load_segmentation_model, predict_segmentation

from depth_estimation import load_depth_model, estimate_depth
from context_builder import ContextBuilder
from speech_Output import text_to_audio, load_kokoro_model
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Global state for recording
class RecordingState:
    def __init__(self):
        self.video_writer = None
        self.audio_frames = []
        self.start_time = None
        self.frame_count = 0


def main():
    st.set_page_config(page_title="SpokenVision", layout="wide")
    st.title("SpokenVision - Real-time Object Detection and Segmentation")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Main Vision", "Settings", "Recording Management"])
    
    with tab1:
        # Initialize session state
        if 'recording_state' not in st.session_state:
            st.session_state.recording_state = RecordingState()
        
        if 'detection_model' not in st.session_state:
            with st.spinner("Loading detection model..."):
                st.session_state.detection_model = load_detection_model()
            st.success("Detection model loaded!")
        
        if 'segmentation_model' not in st.session_state:
            with st.spinner("Loading segmentation model..."):
                st.session_state.segmentation_model, st.session_state.feature_extractor = load_segmentation_model()
            st.success("Segmentation model loaded!")
        
        if 'depth_model' not in st.session_state:
            with st.spinner("Loading depth model..."):
                st.session_state.depth_model = load_depth_model()
            st.success("Depth model loaded!")
        
        # Camera selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
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
        
        with col2:
            record_journey = st.checkbox("Record Journey", key="record_journey")
            if record_journey:
                st.write("🔴 Recording active")
            else:
                st.write("⭕ Not recording")
        
        # Display main video feed
        stframe = st.empty()
        stop_button = st.button("Stop", key="stop_button")
        
        # Recording information
        if record_journey:
            st.info("📹 Your journey is being recorded. Videos and audio are saved in the recordings folder.")
        
        # Start camera feed
        run_camera_feed(camera_choice, camera_options, stop_button, stframe, record_journey)
    
    with tab2:
        st.header("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
        
        st.header("Visualization Settings")
        show_segmentation = st.checkbox("Show Segmentation", True)
        segmentation_opacity = st.slider("Segmentation Opacity", 0.0, 1.0, 0.3, 0.05)
        show_depth = st.checkbox("Show Depth Map", True)
        
        st.header("Audio Settings")
        audio_enabled = st.checkbox("Enable Audio Descriptions", True)
        audio_interval = st.slider("Audio Interval (seconds)", 5, 30, 10, 5)
        
        # Save settings to session state
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.show_segmentation = show_segmentation
        st.session_state.segmentation_opacity = segmentation_opacity
        st.session_state.show_depth = show_depth
        st.session_state.audio_enabled = audio_enabled
        st.session_state.audio_interval = audio_interval
    
    with tab3:
        st.header("Recording Management")
        
        # Show existing recordings
        recordings_dir = "./recordings"
        if os.path.exists(recordings_dir):
            recordings = os.listdir(recordings_dir)
            if recordings:
                st.subheader("Existing Recordings")
                for recording in recordings:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(recording)
                    with col2:
                        if st.button(f"Play {recording}", key=f"play_{recording}"):
                            play_recording(os.path.join(recordings_dir, recording))
                    with col3:
                        if st.button(f"Delete {recording}", key=f"delete_{recording}"):
                            os.remove(os.path.join(recordings_dir, recording))
                            st.rerun()
            else:
                st.write("No recordings found")
        else:
            st.write("No recordings directory found")
        
        # Clear all recordings button
        if st.button("Clear All Recordings"):
            if os.path.exists(recordings_dir):
                for file in os.listdir(recordings_dir):
                    os.remove(os.path.join(recordings_dir, file))
                st.success("All recordings cleared!")
                st.rerun()

def run_camera_feed(camera_choice, camera_options, stop_button, stframe, record_journey):
    """Run the camera feed with all processing"""
    camera_source = camera_options[camera_choice]
    
    # Create output directories
    os.makedirs("./recordings", exist_ok=True)
    os.makedirs("./audio_output", exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_source}")
            return
        
        # Initialize recording
        video_writer = None
        if record_journey:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"./recordings/journey_{timestamp}.mp4"
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        
        # Context builder for audio descriptions
        context_builder = ContextBuilder()
        last_audio_time = time.time()
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Get settings from session state
            confidence_threshold = st.session_state.get('confidence_threshold', 0.3)
            show_segmentation = st.session_state.get('show_segmentation', True)
            segmentation_opacity = st.session_state.get('segmentation_opacity', 0.3)
            show_depth = st.session_state.get('show_depth', True)
            audio_enabled = st.session_state.get('audio_enabled', True)
            audio_interval = st.session_state.get('audio_interval', 10)
            
            # Object detection
            results = detect_objects(frame, st.session_state.detection_model, conf_threshold=confidence_threshold)
            
            blended_frame = frame.copy()
            
            # Depth estimation
            depth_map = None
            if show_depth:
                depth_map = estimate_depth(frame, st.session_state.depth_model)
                depth_map_resized = cv2.resize(depth_map, (frame_width, frame_height))
            
            # Segmentation
            if show_segmentation:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, frame)
                
                segmentation_map = predict_segmentation(
                    temp_file_path,
                    st.session_state.segmentation_model,
                    st.session_state.feature_extractor
                )
                
                segmentation_map_resized = F.interpolate(
                    segmentation_map.unsqueeze(0).unsqueeze(0).float(),
                    size=(frame_height, frame_width),
                    mode='nearest'
                ).squeeze().long()
                
                # Create colored segmentation visualization
                segmentation_color = np.zeros_like(frame)
                num_classes = segmentation_map_resized.max().item() + 1
                colors = {}
                for i in range(num_classes):
                    colors[i] = np.array([np.random.randint(0, 255) for _ in range(3)])
                
                segmentation_np = segmentation_map_resized.cpu().numpy()
                for i in range(num_classes):
                    mask = (segmentation_np == i)
                    segmentation_color[mask] = colors[i]
                
                blended_frame = cv2.addWeighted(
                    frame, 1.0 - segmentation_opacity,
                    segmentation_color, segmentation_opacity, 0
                )
                
                os.remove(temp_file_path)
            
            # Draw object detection results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(blended_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(blended_frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame to video if recording
            if record_journey and video_writer is not None:
                video_writer.write(blended_frame)
            
            # Audio descriptions
            if audio_enabled and time.time() - last_audio_time >= audio_interval:
                generate_audio_description(frame, results, depth_map, segmentation_np if show_segmentation else None)
                last_audio_time = time.time()
            
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)
            
            time.sleep(0.05)
    
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        if video_writer is not None:
            video_writer.release()
        st.write("Camera released")

def generate_audio_description(frame, results, depth_map, segmentation_map):
    """Generate audio description for the current frame"""
    # Here you would integrate your GPT-3.5 enhancement
    # For now, just create a simple description
    description = f"I can see {len(results['labels'])} objects: "
    description += ", ".join(results['labels'])
    
    # Save and play audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_to_audio(
        None,
        description,
        file_name=f"description_{timestamp}",
        tts_backend='gtts',
        voice='en'
    )

def play_recording(file_path):
    """Play a recorded file based on the operating system"""
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(['open', file_path])
    elif platform.system() == 'Windows':
        os.startfile(file_path)
    else:  # Linux
        subprocess.call(['xdg-open', file_path])

if __name__ == "__main__":
    main()
