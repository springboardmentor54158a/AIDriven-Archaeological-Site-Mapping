import json
import os
from PIL import Image

NDJSON_FILE = "labelbox_yolo.ndjson"
IMAGES_DIR = "images"
LABELS_DIR = "labels"

os.makedirs(f"{LABELS_DIR}/train", exist_ok=True)
os.makedirs(f"{LABELS_DIR}/val", exist_ok=True)

def convert_bbox(bbox, img_w, img_h):
    x_center = (bbox["left"] + bbox["width"] / 2) / img_w
    y_center = (bbox["top"] + bbox["height"] / 2) / img_h
    w = bbox["width"] / img_w
    h = bbox["height"] / img_h
    return x_center, y_center, w, h

with open(NDJSON_FILE, "r") as f:
    for line in f:
        data = json.loads(line)

        image_name = data["data_row"]["external_id"]
        split = "train" if image_name in os.listdir(f"{IMAGES_DIR}/train") else "val"

        image_path = f"{IMAGES_DIR}/{split}/{image_name}"
        label_path = f"{LABELS_DIR}/{split}/{image_name.replace('.png', '.txt')}"

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path)
        img_w, img_h = img.size

        annotations = data["projects"][list(data["projects"].keys())[0]]["labels"][0]["annotations"]["objects"]

        with open(label_path, "w") as out:
            for obj in annotations:
                bbox = obj["bounding_box"]
                x, y, w, h = convert_bbox(bbox, img_w, img_h)
                out.write(f"0 {x} {y} {w} {h}\n")

print("YOLO labels generated successfully!")
