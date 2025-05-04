import gradio as gr
import cv2
import numpy as np
import os
import requests
import base64
import base64
import io
import soundfile as sf

# Backend server URL
backend_server_url = "https://0416-2600-1017-a410-36b8-2357-52be-1318-959b.ngrok-free.app"

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

# # Gradio processing function
def process_webcam(image):
    if image is None:
        return None, None

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = send_to_backend(frame)
    print(len(result))
    caption = result.get("caption", "No caption")
    audio_base64 = result.get("audio_base64", None)

    if audio_base64:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        audio_array, sample_rate = sf.read(audio_buffer)
        return caption, (sample_rate, audio_array)

    return caption, None


# Gradio interface
demo = gr.Interface(
    fn=process_webcam,
    inputs=gr.Image(sources=["upload", "webcam"]),    
    outputs=[
        gr.Textbox(label="Caption"),
        gr.Audio(label="Audio Output")
    ],
    live=True,
    title="SpokenVision",
    description="Real-time object detection and captioning with audio feedback",
    allow_flagging="never"
)

demo.launch()
