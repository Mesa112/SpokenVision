from camera import CameraModule
from object_detection import load_model, detect_objects
from semantic_segmentation import load_model as load_segmentation_model
from semantic_segmentation import predict_segmentation, visualize_segmentation
import cv2
import tempfile
import numpy as np
import os
import time
import torch.nn.functional as F
import torch
from dotenv import load_dotenv

from depth_estimation import load_depth_model, estimate_depth
from qwen_captioning import load_qwen_captioning_model, generate_qwen_caption
from context_builder import ContextBuilder
from openai import OpenAI
from speech_Output import text_to_audio, load_kokoro_model


load_dotenv()


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
        system_prompt = """You are an assistive technology that concisely describes the image to a visually impaired person. 
            Convert the technical scene description into natural, concise, helpful speech that would immediately let a visually impaired person understand their environment.
            Instructions:
            - Be informative but concise, aim to orient the listener quickly and effectively.
            - Using simple language and avoiding jargon
            - Using conversational tone
            - Avoiding unnecessary details that may confuse the listener
            - Do not use vague phrases like "in the image" or "just so you know."
            - If the direction isn’t known, omit the object unless it's essential.
            - Describing important objects and their spatial relationships naturally
            - Providing useful context that helps the person navigate and understand their environment
            - Being concise but informative
            - Using directional terms (left, right, in front, behind) that would be helpful for orientation
            - Mentioning any potential hazards or important changes in the scene
            """

        # Call GPT to enhance the description
        response = openai_text_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed to GPT-3.5-Turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this technical description into natural speech: {raw_description}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        enhanced_description = response.choices[0].message.content.strip()
        return enhanced_description
    
    except Exception as e:
        print(f"Error enhancing description with GPT: {e}")
        return raw_description

if __name__ == "__main__":
    # Load all models
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading models...")
    model = load_model().to(device)
    model_segmentation, feature_extractor = load_segmentation_model()
    depth_model = load_depth_model()
    
    try:
        qwen_model = load_qwen_captioning_model()
        use_qwen = True
    except Exception as e:
        print(f"Warning: Could not load Qwen model: {e}")
        use_qwen = False
    
    # Load TTS model (now a simple placeholder)
    kokoro = load_kokoro_model()
    
    # Create context builder
    context_builder = ContextBuilder()
    print("All models loaded successfully.")

    # Initialize camera
    camera_module = CameraModule(camera_index=0, use_depth=False)
    if not camera_module.initialize():
        print("Failed to initialize camera.")
        exit()

    print("Press 'q' to exit.")
    frame_counter = 1 
    last_audio_time = time.time()
    last_processed_frame_time = time.time()
    audio_interval = 10.0  # Generate speech every 10 seconds
    process_interval = 0.5  # Process frames every 0.5 seconds (2 FPS)
    
    # For frame skipping
    frame_skip_count = 0
    frame_skip_rate = 14  # Process 1 frame every 15 frames at 30 FPS
    
    context_description = ""  # To store the latest context
    raw_context_description = ""  # To store the original technical description
    
    # Store the latest processed data for visualization
    latest_segmentation_map = None
    latest_results = None
    latest_depth_map = None

    # Always use gTTS for speech output
    tts_backend = 'gtts'
    tts_voice = 'en'

    try:
        while True:
            # Capture frame
            frame, _ = camera_module.get_frame()
            if frame is None:
                print("Failed to capture frame.")
                continue
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Only process frames at specified interval
            current_time = time.time()
            frame_skip_count += 1
            
            if current_time - last_processed_frame_time >= process_interval or frame_skip_count >= frame_skip_rate:
                frame_skip_count = 0
                last_processed_frame_time = current_time
                
                # Run object detection
                latest_results = detect_objects(frame, model, conf_threshold=0.3)

                # Run depth estimation
                depth_map = estimate_depth(frame, depth_model)
                latest_depth_map = cv2.resize(depth_map, (frame_width, frame_height))

                # Save frame for segmentation
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, frame)

                # Run semantic segmentation
                segmentation_map = predict_segmentation(temp_file_path, model_segmentation, feature_extractor)

                # Resize segmentation map to match frame dimensions
                latest_segmentation_map = F.interpolate(
                    segmentation_map.unsqueeze(0).unsqueeze(0).float(),
                    size=(frame_height, frame_width),
                    mode='nearest'
                ).squeeze().long()

                # Generate image caption from Qwen if available
                qwen_caption = ""
                if use_qwen:
                    try:
                        qwen_caption = generate_qwen_caption(frame, qwen_model)[0]
                    except Exception as e:
                        print(f"Warning: Error generating Qwen caption: {e}")
                
                # Build rich context from all model outputs
                raw_context_description = context_builder.process_frame_data(
                    latest_results, 
                    latest_depth_map, 
                    latest_segmentation_map.cpu().numpy(), 
                    qwen_caption
                )
                
                # Enhance the description with GPT for natural speech
                context_description = enhance_description_with_gpt(raw_context_description)
                
                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
            # Always show the latest visualization if we have data
            if latest_segmentation_map is not None and latest_results is not None and latest_depth_map is not None:
                # Create visualization
                segmentation_color = np.zeros_like(frame)
                num_classes = latest_segmentation_map.max().item() + 1
                colors = {}
                for i in range(num_classes):
                    colors[i] = np.array([np.random.randint(0, 256) for _ in range(3)])

                segmentation_np = latest_segmentation_map.cpu().numpy()
                for i in range(num_classes):
                    mask = (segmentation_np == i)
                    segmentation_color[mask] = colors[i]

                # Blend original frame with segmentation
                blended_frame = cv2.addWeighted(frame, 0.7, segmentation_color, 0.3, 0)

                # Draw object detection boxes
                for score, label, box in zip(latest_results["scores"], latest_results["labels"], latest_results["boxes"]):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(blended_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(blended_frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display visualization windows
                cv2.imshow("SpokenVision Detection & Segmentation", blended_frame)
                cv2.imshow("Depth Map", latest_depth_map)
            
            # Generate speech at intervals
            if current_time - last_audio_time >= audio_interval and context_description:
                # Create output directory if it doesn't exist
                os.makedirs("./audio_output", exist_ok=True)
                
                # Generate speech using gTTS with enhanced description
                file_name = f"Frame_{frame_counter}"
                
                try:
                    text_to_audio(
                        kokoro, 
                        context_description, 
                        file_name=file_name,
                        tts_backend=tts_backend,
                        voice=tts_voice
                    )
                except Exception as e:
                    print(f"Error with gTTS: {e}")
                
                print(f"\n========== RAW CONTEXT DESCRIPTION ==========")
                print(raw_context_description)
                print("\n========== ENHANCED DESCRIPTION ==========")
                print(context_description)
                print("=======================================\n")
                
                last_audio_time = current_time
                frame_counter += 1
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    finally:
        camera_module.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")