from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import torch

def load_model():
    """
    Load a publicly available segmentation model.
    Returns:
    model: Loaded segmentation model
    feature_extractor: Feature extractor for preprocessing images
    """
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

def preprocess_image(image_path, feature_extractor):
    """
    Preprocess the image for the model.
    Args:
    image_path: Path to the input image
    feature_extractor: Feature extractor for preprocessing images
    Returns:
    pixel_values: Preprocessed image tensor
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def predict_segmentation(image_path, model, feature_extractor):
    """
    Predict the segmentation map for the input image.
    Args:
    image_path: Path to the input image
    model: Loaded segmentation model
    feature_extractor: Feature extractor for preprocessing images
    Returns:
    segmentation_map: Predicted segmentation map
    """
    inputs = preprocess_image(image_path, feature_extractor)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch_size, num_classes, height, width)
        
    # Get the predicted class for each pixel
    segmentation_map = logits.argmax(dim=1)[0]
    return segmentation_map

def visualize_segmentation(image_path, segmentation_map):
    """
    Visualize the segmentation map.
    Args:
    image_path: Path to the input image
    segmentation_map: Predicted segmentation map
    """
    image = Image.open(image_path)
    segmentation_map_np = segmentation_map.byte().cpu().numpy()
    
    segmentation_map_pil = Image.fromarray(segmentation_map_np)
    segmentation_map_pil.putpalette([i for _ in range(256) for i in range(256)]*3)
    
    image.show()
    segmentation_map_pil.show()
