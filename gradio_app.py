import gradio as gr
import cv2
import tempfile
import numpy as np
import os
import time
import requests
import base64
import threading
import pygame

# Backend server URL
backend_server_url = "https://0416-2600-1017-a410-36b8-2357-52be-1318-959b.ngrok-free.app"

send_thread = None # To keep track of ongoing threads

# Audio playback
def play_audio(audio_base64):
    """
    Play audio file using pygame mixer.
    
    Args:
        audio_path: Path to audio file
    """
    audio_bytes = base64.b64decode(audio_base64)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Audio error: {e}")

# Backend interaction
def send_to_backend(frame):
    try:
        # _, img_encoded = cv2.imencode('.jpg', frame)
        # img_bytes = img_encoded.tobytes()
        small_frame = cv2.resize(frame, (224, 224))  
        # Save current frame to disk
        cv2.imwrite("frame.jpg", small_frame)

        # Ensure dummy audio file exists
        empty_audio_path = "input.mp3"
        if not os.path.exists(empty_audio_path):
            with open(empty_audio_path, "wb") as f:
                f.write(b"")

        with open("frame.jpg", "rb") as img, open("input.mp3", "rb") as audio:
            files = {
                "image": ("frame.jpg", img, "image/jpeg"),
                "audio": ("input.mp3", audio, "audio/mpeg")
            }
            response = requests.post(backend_server_url + "/process/", files=files)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Backend error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

def thread_sendToBackend(frame):
    """ Starts a thread to send the frame to the backend. """
    global send_thread
    if send_thread is None:
        send_thread = threading.Thread(target=send_to_backend, args=(frame,), daemon=True)
        send_thread.start()


# Gradio processing function
def process_webcam(image):
    if image is None:
        return None, "No frame", None

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = send_to_backend(frame)
    print(len(result))
    caption = result['caption']
    audio_base64 = result['audio_base64']

    if audio_base64:
        threading.Thread(target=play_audio, args=(audio_base64,), daemon=True).start()

    return caption


# Gradio interface
demo = gr.Interface(
    fn=process_webcam,
    inputs=gr.Image(sources=["webcam"]),#[gr.Image(sources=["webcam"]), gr.Video(sources=["webcam"])],
    outputs=[
        gr.Textbox(label="Caption"),
    ],
    live=True,
    title="SpokenVision",
    description="Real-time object detection and captioning with audio feedback"
)

demo.launch()
