from ultralytics import YOLO

model = YOLO('yolov8n.pt')
class_names = model.names
for i, name in class_names.items():
    print(f"{i}: {name}")
