from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

qwen_processor = None

def load_qwen_captioning_model():
    global qwen_processor 
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True
    )
    # No tengo Nvidia GPU
    # qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    qwen_model.eval()
    return qwen_model

def generate_qwen_caption(frame, qwen_model, input=""):
    if input == "":
        text = """
            You are an assistive technology that describes an image to a visually impaired 
            person in a clear, helpful, and natural way.

            Your task:
            Convert a technical scene description into concise, 
            easy-to-understand speech that helps the listener quickly 
            understand their environment.

            Instructions:

            Start by describing what’s closest to the listener first, 
            then move outward or deeper into the space.

            Immediately identify and locate people or key objects 
            in relation to the listener (e.g., “A person is standing directly in front of you…”).

            Be specific about where things are using natural 
            spatial language (left, right, behind, on the wall to your left, etc.).

            Do not say “in the image,” “just so you know,” 
            “it looks like,” or other filler phrases. Get straight to the point.

            Clearly describe walls, doors, furniture, 
            or objects with reference to their positions.

            Mention anything a person might want to know 
            for awareness or navigation—like obstacles or changes in the scene.

            Keep the tone natural but focused. 
            Use short, informative sentences.
        """
    else:
        text = f"""
            You are an assistive technology that describes an image to a visually impaired 
            person in a clear, helpful, and natural way.

            Focus on the user request: {input}

            Your task:
            Convert a technical scene description into concise, 
            easy-to-understand speech that helps the listener quickly 
            understand their environment.

            Instructions:
            Answer the user's request directly and clearly first. Then, if relevant, 
            describe the rest of the scene from closest to furthest. 
            Only describe what’s necessary to help the user. 
            Do not add unrelated details.   

            Keep the tone natural but focused. 
            Use short, informative sentences.
        """
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    # Convert the frame to a PIL image
    pil_image = Image.fromarray(frame)

    # Preprocess the image and prepare the inputs for the model
    text_prompt = qwen_processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = qwen_processor(
    text=[text_prompt], images=[pil_image], padding=True, return_tensors="pt"
    )
    # Generate the caption using the model
    output_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    caption = qwen_processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return caption