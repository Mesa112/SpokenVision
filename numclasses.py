from ultralytics import YOLO

from ultralytics import YOLO

# Load the pretrained YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, etc.)
model = YOLO('yolov8n.pt')

# Get the class names
class_names = model.names

# Print the class names
for i, name in class_names.items():
    print(f"{i}: {name}")
