"""
Convert polygon annotations from Labelbox to YOLOv5 format (bounding boxes)
YOLOv5 format: class_id center_x center_y width height (normalized 0-1)
"""
import json
import os
from pathlib import Path

# Class mapping: class_id for YOLOv5 (0-indexed, excluding background)
CLASS_MAP = {
    "vegitation": 0,     # vegetation class (ID 0)
    "vegetation": 0,     # alternative spelling
    "ruins": 1,          # ruins class (ID 1)
    # background is not included in object detection
}

def polygon_to_bbox(polygon):
    """Convert polygon to bounding box [x_min, y_min, x_max, y_max]"""
    xs = [p["x"] for p in polygon]
    ys = [p["y"] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert bbox [x_min, y_min, x_max, y_max] to YOLO format [center_x, center_y, width, height] (normalized)"""
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center and dimensions
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return [center_x, center_y, width, height]

def convert_annotations(ndjson_file, images_dir, output_labels_dir):
    """Convert Labelbox NDJSON annotations to YOLOv5 format"""
    os.makedirs(output_labels_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    with open(ndjson_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            img_name = data["data_row"]["external_id"]
            img_path = os.path.join(images_dir, img_name)
            
            # Handle .jpg / .png mismatch
            if not os.path.exists(img_path):
                if img_path.endswith(".png"):
                    img_path = img_path.replace(".png", ".jpg")
                else:
                    img_path = img_path.replace(".jpg", ".png")
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_name}")
                skipped += 1
                continue
            
            # Get image dimensions
            from PIL import Image
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Collect all bounding boxes
            yolo_annotations = []
            
            for project in data.get("projects", {}).values():
                for label in project.get("labels", []):
                    for obj in label.get("annotations", {}).get("objects", []):
                        label_name = obj["value"].lower()
                        
                        if label_name not in CLASS_MAP:
                            print(f"Warning: Unknown label: {label_name} in {img_name}")
                            continue
                        
                        class_id = CLASS_MAP[label_name]
                        polygon = obj.get("polygon", [])
                        
                        if len(polygon) < 3:
                            continue
                        
                        # Convert polygon to bounding box
                        bbox = polygon_to_bbox(polygon)
                        yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
                        
                        # YOLO format: class_id center_x center_y width height
                        yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            
            # Write YOLO format label file
            base_name = os.path.splitext(img_name)[0]
            label_file = os.path.join(output_labels_dir, base_name + ".txt")
            
            with open(label_file, 'w') as lbl_f:
                lbl_f.writelines(yolo_annotations)
            
            if yolo_annotations:
                processed += 1
                print(f"Converted {img_name}: {len(yolo_annotations)} objects")
            else:
                skipped += 1
                print(f"Warning: No objects found in {img_name}")
    
    print(f"\nConversion complete!")
    print(f"   Processed: {processed} images")
    print(f"   Skipped: {skipped} images")

if __name__ == "__main__":
    import sys
    
    ndjson_file = "labelbox_export2.ndjson"
    images_dir = "images"
    output_labels_dir = "labels"
    
    if len(sys.argv) > 1:
        ndjson_file = sys.argv[1]
    if len(sys.argv) > 2:
        images_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_labels_dir = sys.argv[3]
    
    convert_annotations(ndjson_file, images_dir, output_labels_dir)

