from camera import CameraModule
from object_detection import load_model, detect_objects
from semantic_segmentation import load_model as load_segmentation_model
from semantic_segmentation import predict_segmentation, visualize_segmentation
import cv2
import tempfile
import numpy as np
import os
import torch.nn.functional as F

from depth_estimation import load_depth_model, estimate_depth
from blip_image_captioning import load_blip_captioning_model, generate_caption
from qwen_captioning import load_qwen_captioning_model, generate_qwen_caption

if __name__ == "__main__":
    model = load_model()
    model_segmentation, feature_extractor = load_segmentation_model()
    depth_model = load_depth_model()
    blip_model = load_blip_captioning_model()
    qwen_model = load_qwen_captioning_model()
    print("Models loaded successfully.")

    camera_module = CameraModule(camera_index=0, use_depth=False)
    if not camera_module.initialize():
        print("Failed to initialize camera.")
        exit()

    print("Press 'q' to exit.")

    try:
        while True:
            #this is the main loop, it will run until the user presses 'q'
            #it will capture frames from the camera and perform object detection
            frame, _ = camera_module.get_frame()

            if frame is None:
                print("Failed to capture frame.")
                continue
            # frame height and width
            frame_height, frame_width = frame.shape[:2]
            #obj detector
            results = detect_objects(frame, model, conf_threshold=0.3)

            #depth estimation (midas)
            depth_map = estimate_depth(frame, depth_model)
            # normalize and resize depth_map to match the frame size
            depth_map_resized = cv2.resize(depth_map, (frame_width, frame_height))

            #save frame to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file_path = temp_file.name
                cv2.imwrite(temp_file_path, frame)

            # Perform semantic segmentation
            segmentation_map = predict_segmentation(temp_file_path, model_segmentation, feature_extractor)

            # Resize segmentation map to match frame dimensions
            segmentation_map_resized = F.interpolate(
                segmentation_map.unsqueeze(0).unsqueeze(0).float(),
                size=(frame_height, frame_width),
                mode='nearest'
            ).squeeze().long()

            segmentation_color = np.zeros_like(frame)
            num_classes = segmentation_map_resized.max().item() + 1
            colors = {}
            for i in range(num_classes):

                colors[i] = np.array([np.random.randint(0, 256) for _ in range(3)])

            segmentation_np = segmentation_map_resized.cpu().numpy()
            for i in range(num_classes):
                mask = (segmentation_np == i)
                segmentation_color[mask] = colors[i]

            blended_frame = cv2.addWeighted(frame, 0.5, segmentation_color, 0.5, 0)

            # Draw results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(blended_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(blended_frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the frame with detections
            cv2.imshow("Live YOLOv8 Detection", blended_frame)
            cv2.imshow("Depth Map (Grayscale)", depth_map_resized)

            print("========Outputs========")
            print("Caption:", generate_caption(frame, blip_model))
            print("Qwen Caption:", generate_qwen_caption(frame, qwen_model))
            #Print object detection + depth results
            print("Object detection:")
            for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Clamp coordinates to be within depth_map size
                x1 = max(0, min(x1, depth_map.shape[1] - 1))
                x2 = max(0, min(x2, depth_map.shape[1] - 1))
                y1 = max(0, min(y1, depth_map.shape[0] - 1))
                y2 = max(0, min(y2, depth_map.shape[0] - 1))

                # Slice the depth map for the region inside the box
                object_depth = depth_map[y1:y2, x1:x2]

                if object_depth.size > 0:
                    average_depth = object_depth.mean()
                    print(f"{label} ({score * 100:.2f}%) at [{x1}, {y1}, {x2}, {y2}] with depth {average_depth:.2f}")
                else:
                    print(f"{label} out of bounds")
            print("========================")

            try:
                os.remove(temp_file_path)
            except:
                pass
                
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        camera_module.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")
