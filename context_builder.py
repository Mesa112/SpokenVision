import numpy as np
import cv2
from collections import defaultdict

class ContextBuilder:
    def __init__(self, proximity_threshold=0.2, depth_cutoffs=[0.3, 0.6, 0.9]):
        """
        Initialize the context builder with configurable parameters.
        
        Args:
            proximity_threshold: Threshold to determine if objects are nearby (as fraction of image size)
            depth_cutoffs: Depth ranges for categorizing objects as [close, medium, far] (normalized values)
        """
        self.proximity_threshold = proximity_threshold
        self.depth_cutoffs = depth_cutoffs
        # Tracking previous frames data
        self.previous_objects = []
        self.new_objects = []
        self.frame_count = 0
        
    def process_frame_data(self, detection_results, depth_map, segmentation_map, caption):
        """
        Process all data from a single frame to build a rich context description.
        
        Args:
            detection_results: Dict with boxes, labels, scores from object detection
            depth_map: Normalized depth map (0-1 values)
            segmentation_map: Semantic segmentation map
            caption: Image caption from BLIP/Qwen
            
        Returns:
            context_description: A structured description of the scene ready for speech
        """
        self.frame_count += 1
        
        # Extract core data
        objects_with_positions = self._process_objects(detection_results, depth_map)
        spatial_relationships = self._analyze_spatial_relationships(objects_with_positions)
        scene_description = self._generate_scene_description(
            objects_with_positions, 
            spatial_relationships, 
            caption
        )
        
        return scene_description
    
    def _process_objects(self, detection_results, depth_map):
        """
        Enrich object detection results with depth and position information.
        
        Returns:
            List of dicts with object data including position and depth category
        """
        objects = []
        
        for box, label, score in zip(
            detection_results["boxes"], 
            detection_results["labels"], 
            detection_results["scores"]
        ):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Calculate center position and size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            size = width * height
            
            # Extract depth for this object
            try:
                # Ensure coordinates are valid for depth map
                x1_valid = max(0, min(x1, depth_map.shape[1] - 1))
                x2_valid = max(0, min(x2, depth_map.shape[1] - 1))
                y1_valid = max(0, min(y1, depth_map.shape[0] - 1))
                y2_valid = max(0, min(y2, depth_map.shape[0] - 1))
                
                # Get average depth in object region
                object_depth_region = depth_map[y1_valid:y2_valid, x1_valid:x2_valid]
                if object_depth_region.size > 0:
                    avg_depth = object_depth_region.mean() / 255.0  # Normalize to 0-1
                else:
                    avg_depth = 0.5  # Default if region is invalid
            except Exception as e:
                print(f"Error getting depth for object: {e}")
                avg_depth = 0.5
            
            # Categorize depth
            if avg_depth < self.depth_cutoffs[0]:
                depth_category = "very close"
            elif avg_depth < self.depth_cutoffs[1]:
                depth_category = "close"
            elif avg_depth < self.depth_cutoffs[2]:
                depth_category = "medium distance"
            else:
                depth_category = "far away"
                
            # Determine position in frame (using 3x3 grid)
            frame_height, frame_width = depth_map.shape[:2]
            
            # Horizontal position
            if center_x < frame_width / 3:
                horiz_pos = "left"
            elif center_x < 2 * frame_width / 3:
                horiz_pos = "center"
            else:
                horiz_pos = "right"
                
            # Vertical position
            if center_y < frame_height / 3:
                vert_pos = "top"
            elif center_y < 2 * frame_height / 3:
                vert_pos = "middle"
            else:
                vert_pos = "bottom"
                
            position = f"{vert_pos} {horiz_pos}"
            
            # Track if this is a new object
            is_new = self._is_new_object(label, (center_x, center_y))
            
            objects.append({
                "label": label,
                "score": score,
                "box": box,
                "center": (center_x, center_y),
                "depth": avg_depth,
                "depth_category": depth_category,
                "position": position,
                "size": size,
                "is_new": is_new
            })
        
        # Update tracking
        self._update_tracked_objects(objects)
        
        return objects
    
    def _is_new_object(self, label, center):
        """Check if this object wasn't in previous frames"""
        if self.frame_count <= 15:  # First few frames, don't consider new
            return False
            
        for prev_obj in self.previous_objects:
            if prev_obj["label"] == label:
                # Check if positions are similar
                prev_center = prev_obj["center"]
                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                if dist < 50:  # Threshold for considering same object
                    return False
        
        return True
    
    def _update_tracked_objects(self, current_objects):
        """Update tracking lists for next frame comparison"""
        self.previous_objects = current_objects
    
    def _analyze_spatial_relationships(self, objects):
        """
        Analyze spatial relationships between objects.
        
        Returns:
            Dict mapping relationship types to lists of object pairs
        """
        relationships = defaultdict(list)
        
        # Skip if too few objects
        if len(objects) < 2:
            return relationships
            
        # Compare each pair of objects
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calculate distance between centers
                center1 = obj1["center"]
                center2 = obj2["center"]
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Check for proximity
                if distance < self.proximity_threshold * max(obj1["box"][2]-obj1["box"][0], obj2["box"][2]-obj2["box"][0]):
                    relationships["nearby"].append((obj1["label"], obj2["label"]))
                
                # Check for alignment
                if abs(center1[1] - center2[1]) < 30:  # Small vertical difference
                    if center1[0] < center2[0]:
                        relationships["side_by_side"].append((obj1["label"], "left of", obj2["label"]))
                    else:
                        relationships["side_by_side"].append((obj2["label"], "left of", obj1["label"]))
                
                # Check if one object is in front of another
                depth_diff = abs(obj1["depth"] - obj2["depth"])
                if depth_diff > 0.2:  # Significant depth difference
                    if obj1["depth"] < obj2["depth"]:
                        relationships["depth_order"].append((obj1["label"], "in front of", obj2["label"]))
                    else:
                        relationships["depth_order"].append((obj2["label"], "in front of", obj1["label"]))
        
        return relationships
    
    def _generate_scene_description(self, objects, relationships, caption):
        """
        Generate a natural language description of the scene.
        
        Returns:
            str: Natural language description ready for speech output
        """
        # Start with the overall caption
        description = f"{caption}"
        
        # Add details about important objects
        if objects:
            # Sort objects by size and confidence to focus on the most important ones
            sorted_objects = sorted(objects, key=lambda x: x["size"] * x["score"], reverse=True)
            
            # Limit to 3-5 most important objects to avoid information overload
            main_objects = sorted_objects[:min(4, len(sorted_objects))]
            
            # Describe new objects first
            new_objects = [obj for obj in main_objects if obj["is_new"]]
            if new_objects:
                description += "\n\nNew in view: " + ", ".join([f"{obj['label']} ({obj['position']})" for obj in new_objects])
            
            # Add details about the most prominent object
            if main_objects:
                primary = main_objects[0]
                description += f"\n\nPrimary object: {primary['label']} at {primary['position']}, {primary['depth_category']}."
                
                # Add details about secondary objects
                if len(main_objects) > 1:
                    secondary_desc = "\nAlso visible: " + ", ".join([
                        f"{obj['label']} ({obj['position']}, {obj['depth_category']})" 
                        for obj in main_objects[1:]
                    ])
                    description += secondary_desc
        
        # Add spatial relationships if available
        if relationships:
            relation_texts = []
            
            if relationships["nearby"]:
                # Limit to 2 most interesting nearby relationships
                nearby = relationships["nearby"][:min(2, len(relationships["nearby"]))]
                for obj1, obj2 in nearby:
                    relation_texts.append(f"{obj1} is near {obj2}")
            
            if relationships["side_by_side"]:
                # Limit to 2 most interesting side-by-side relationships
                side_by_side = relationships["side_by_side"][:min(2, len(relationships["side_by_side"]))]
                for obj1, relation, obj2 in side_by_side:
                    relation_texts.append(f"{obj1} is {relation} {obj2}")
            
            if relationships["depth_order"]:
                # Limit to 2 most interesting depth relationships
                depth_order = relationships["depth_order"][:min(2, len(relationships["depth_order"]))]
                for obj1, relation, obj2 in depth_order:
                    relation_texts.append(f"{obj1} is {relation} {obj2}")
            
            if relation_texts:
                description += "\n\nSpatial information: " + ". ".join(relation_texts)
        
        return description


def integrate_with_main(frame, detection_results, depth_map, segmentation_map, caption):
    """
    Helper function to integrate the context builder with the main.py file.
    
    Example usage in main.py:
    ```
    from context_builder import integrate_with_main, ContextBuilder
    
    # Initialize context builder (once at the beginning)
    context_builder = ContextBuilder()
    
    # In the main loop:
    context_description = integrate_with_main(
        frame, results, depth_map, segmentation_map_resized.cpu().numpy(), caption
    )
    print("Context Description:", context_description)
    ```
    """
    # Create context builder if one doesn't exist
    if not hasattr(integrate_with_main, "context_builder"):
        integrate_with_main.context_builder = ContextBuilder()
    
    # Process the frame data
    context_description = integrate_with_main.context_builder.process_frame_data(
        detection_results, depth_map, segmentation_map, caption
    )
    
    return context_description