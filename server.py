from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tempfile
import cv2
import os
import tempfile
from typing import Optional
import base64
import torch.nn.functional as F
from openai import OpenAI
from dotenv import load_dotenv

from object_detection import load_model, detect_objects
from semantic_segmentation import load_model as load_segmentation_model, predict_segmentation
from depth_estimation import load_depth_model, estimate_depth
from blip_image_captioning import load_blip_captioning_model, generate_caption
from kokoro_audio import text_to_audio, load_kokoro_model
from qwen_captioning import load_qwen_captioning_model, generate_qwen_caption
from context_builder import ContextBuilder

from PIL import Image
import io

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # or ["*"] for all, but not recommended in prod. Set to hosted url later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# Load models on startup
model = load_model()
segmentation_model, feature_extractor = load_segmentation_model()
depth_model = load_depth_model()
blip_model = load_blip_captioning_model()
kokoro_model = load_kokoro_model()
context_builder = ContextBuilder()

try:
    qwen_model = load_qwen_captioning_model()
    use_qwen = True
except Exception as e:
    print(f"Warning: Could not load Qwen model: {e}")
    use_qwen = False

print("Models loaded successfully.")

openai_text_client = None 
try:
    openai_text_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_chat = openai_text_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("OpenAI text processing is available.")
except Exception as e:
    print(f"OpenAI text processing not available: {e}")
    openai_text_client = None

def enhance_description_with_gpt(raw_description):
    """Use GPT-3.5-Turbo to convert technical descriptions into natural, conversational speech."""
    if not openai_text_client:
        return raw_description
    
    try:
        # Create a system prompt for natural speech conversion
        system_prompt = """
        You are an assistive technology describing visual scenes to a blind user. Your goal is to give a clear, short, spoken description of their surroundings that will help them navigate in real-time.

        Guidelines:

        Focus on the closest and most relevant objects first.

        Provide a quick, general description of the area after the key objects.

        Use clear spatial language (e.g., "to your left," "in front of you," "on the table").

        Avoid filler phrases or long descriptions.

        Keep it brief and direct to allow for fast understanding.

        The description should help the listener get a quick grasp of their surroundings without overwhelming them. Prioritize the most important objects and hazards.

        """
        # Call GPT to enhance the description
        response = openai_text_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed to GPT-3.5-Turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{raw_description}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        enhanced_description = response.choices[0].message.content.strip()
        return enhanced_description
    
    except Exception as e:
        print(f"Error enhancing description with GPT: {e}")
        return raw_description
    
### 
###
### Endpoint to process image and audio files
@app.post("/process/")
async def process_files(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None)  # Optional audio file
):
    print("Received input files.")
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # NEED TO CONVERT TO RGB!! or else image will have a blue hue
        frame_height, frame_width = frame.shape[:2] #Frame dimensions

        # Run object detection
        latest_results = detect_objects(frame, model, conf_threshold=0.3)

        # Run depth estimation
        depth_map = estimate_depth(frame, depth_model)
        latest_depth_map = cv2.resize(depth_map, (frame_width, frame_height))

        print("Object detection and depth estimation completed.")
        # Save frame for segmentation
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, frame)
        # Run semantic segmentation
        segmentation_map = predict_segmentation(temp_file_path, segmentation_model, feature_extractor)

        # Resize segmentation map to match frame dimensions
        latest_segmentation_map = F.interpolate(
            segmentation_map.unsqueeze(0).unsqueeze(0).float(),
            size=(frame_height, frame_width),
            mode='nearest'
        ).squeeze().long()

        print("Semantic segmentation completed.")

        # Generate caption
        if use_qwen:
            caption = generate_qwen_caption(frame, qwen_model)[0]
        else:
            caption = generate_caption(frame, blip_model)

        print("Caption generation completed.")

        # Process context
        raw_context_description = context_builder.process_frame_data(
            latest_results, 
            latest_depth_map, 
            latest_segmentation_map.cpu().numpy(), 
            caption
        )
        print("Context processing completed.")

        # Enhance the description with GPT for natural speech
        caption = enhance_description_with_gpt(raw_context_description)
        print("Caption enhancement completed.")


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