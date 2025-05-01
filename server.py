from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np
import tempfile
import cv2
import os
import tempfile
from typing import Optional
import base64

from object_detection import load_model, detect_objects
from semantic_segmentation import load_model as load_segmentation_model, predict_segmentation
from depth_estimation import load_depth_model, estimate_depth
from blip_image_captioning import load_blip_captioning_model, generate_caption
from kokoro_audio import text_to_audio, load_kokoro_model

# Hosting the server locally:
# uvicorn server:app --host 0.0.0.0 --port 8000
# ngrok http 8000 (seprate terminal, for public hosting. Need to install ngrok first)

# Test with curl command (client side)
# curl.exe -X POST "http://localhost:8000/process/" -F "image=@<local image file>" -F "audio=@<Local mp3 audio file>" --output output_audio.wav
    # if using ngrok, replace localhost with the ngrok URL.
    # replace <local image file> and <Local mp3 audio file> with the paths to your files.
    # audio is optional
    #output audio will be saved in the current directory as output_audio.wav



app = FastAPI()

# Load models on startup
model = load_model()
segmentation_model, feature_extractor = load_segmentation_model()
depth_model = load_depth_model()
blip_model = load_blip_captioning_model()
kokoro_model = load_kokoro_model()
print("Models loaded successfully.")

@app.post("/process/")
async def process_files(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None)  # Optional audio file
):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Generate caption
        caption = generate_caption(frame, blip_model)

        if audio is not None: #same code as the else statement but later can modify for audio->text and input into captioning 
            print("Audio file received.")
            # Get the system temp directory
            temp_dir = tempfile.gettempdir()

            audio_output_dir = os.path.join(temp_dir, "audio_output")
            # Create the directory if it doesn't exist
            if not os.path.exists(audio_output_dir):
                os.makedirs(audio_output_dir)

            # Converts caption to audio using Kokoro
            text_to_audio(kokoro_model, caption, output_dir=audio_output_dir)

            # audio output file path
            wav_output_path = os.path.join(audio_output_dir, "audio_output.wav")
            with open(wav_output_path, "rb") as f:
                audio_bytes = f.read()
                encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

            return JSONResponse(content={
                "caption": caption,
                "audio_base64": encoded_audio
            })
        else:
            temp_dir = tempfile.gettempdir()

            audio_output_dir = os.path.join(temp_dir, "audio_output")
            # Create the directory if it doesn't exist
            if not os.path.exists(audio_output_dir):
                os.makedirs(audio_output_dir)

            # Converts caption to audio using Kokoro
            text_to_audio(kokoro_model, caption, output_dir=audio_output_dir)

            # audio output file path
            wav_output_path = os.path.join(audio_output_dir, "audio_output.wav")
            with open(wav_output_path, "rb") as f:
                audio_bytes = f.read()
                encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

            return JSONResponse(content={
                "caption": caption,
                "audio_base64": encoded_audio
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
# @app.get("/audio")
# def get_audio():
#     audio_path = f"{tempfile.gettempdir()}/caption_audio.wav"
#     if os.path.exists(audio_path):
#         return FileResponse(audio_path, media_type="audio/wav", filename="caption_audio.wav")
#     return JSONResponse(status_code=404, content={"error": "Audio not found."})

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)