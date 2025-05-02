import streamlit as st
import cv2
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
import time
import requests
import json
import base64
import threading


import queue
response_queue = queue.Queue() #For thread-safe communication between threads

def playAudio(audio_base64):
    # Decode the base64 string into bytes
    audio_bytes = base64.b64decode(audio_base64)

    # Save to a file
    audio_path = "output_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Display audio player in Streamlit
    st.audio(audio_bytes, format="audio/wav")

def sendToBackend(frame, audio = None):
    try:
        # Save current frame to disk
        cv2.imwrite("frame.jpg", frame)

        # Create an empty audio file (1 second of silence if needed)
        empty_audio_path = "input.mp3"
        if not os.path.exists(empty_audio_path):
            with open(empty_audio_path, "wb") as f:
                f.write(b"")

        with open("frame.jpg", "rb") as img, open("input.mp3", "rb") as audio:
            files = {
                "image": ("frame.jpg", img, "image/jpeg"),
                "audio": ("input.mp3", audio, "audio/mpeg")
            }

            #response = requests.post("http://localhost:8000/process/", files=files)
            response = requests.post("https://8804-2600-1017-a410-36b8-2357-52be-1318-959b.ngrok-free.app/process/", files=files)
            
            if response.status_code == 200: #If the request was successful
                st.success("Frame sent successfully!")
                response_queue.put(response.json())
            else:
                st.error(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error sending frame: {e}")

def main():
    st.title("Real-time Object Detection and Segmentation")
    
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
    
    # Camera stream capture
    stframe = st.empty()
    camera_source = camera_options[camera_choice]
    
    # Setup capture once and keep it in session
    if 'cap' not in st.session_state:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_source}")
        else:
            st.session_state.cap = cap
            st.session_state.streaming = True
            st.session_state.paused = False
            st.session_state.last_frame = None
    else:
        cap = st.session_state.cap


    # Initialize session state keys
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    if 'toggle_pressed' not in st.session_state:
        st.session_state.toggle_pressed = False

    def toggle_pause():
        st.session_state.paused = not st.session_state.paused
        st.session_state.toggle_pressed = True


    # Button logic
    col1, col2 = st.columns(2)
    with col1:
        st.button("Resume" if st.session_state.paused else "Pause", on_click=toggle_pause)

    with col2:
        if st.button("Send Frame") and st.session_state.last_frame is not None:
            threading.Thread(target=sendToBackend, args=(st.session_state.last_frame,)).start()
            #sendToBackend(st.session_state.last_frame)

    # Start camera feed
    try:
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_source}")
            return
        
        while True:
            if not st.session_state.paused:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from camera")
                    break
                st.session_state.last_frame = frame  # Save last good frame
            else:
                frame = st.session_state.get('last_frame', None)
                if frame is None:
                    time.sleep(0.05)
                    continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame in Streamlit
            stframe.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Display response if available
            if not response_queue.empty():
                response = response_queue.get()
                st.markdown("### Server Response")

                if "caption" in response:
                    st.write("Caption:", response["caption"])

                if "audio_base64" in response:
                    playAudio(response["audio_base64"])
                
            # Small delay to prevent high CPU usage
            time.sleep(0.05)
            
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()