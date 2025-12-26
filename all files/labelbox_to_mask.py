import json
import os
import numpy as np
import cv2
from PIL import Image

# ===== PATHS =====
NDJSON_PATH = "labelbox_export.ndjson"
IMAGES_DIR = "images"
MASKS_DIR = "masks"

os.makedirs(MASKS_DIR, exist_ok=True)

# ===== CLASS MAPPING =====
CLASS_MAP = {
    "Ruins": 1,
    "Vegetation": 2,
    "Terrain": 3
}

# ===== READ NDJSON =====
data = []
with open(NDJSON_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# ===== PROCESS EACH IMAGE =====
for item in data:
    external_id = item["data_row"]["external_id"]
    image_path = os.path.join(IMAGES_DIR, external_id)

    if not os.path.exists(image_path):
        print(f"Image not found: {external_id}")
        continue

    image = Image.open(image_path)
    width, height = image.size

    mask = np.zeros((height, width), dtype=np.uint8)

    for project in item["projects"].values():
        for label in project["labels"]:
            objects = label["annotations"]["objects"]

            for obj in objects:
                class_name = obj["name"]
                if class_name not in CLASS_MAP:
                    continue

                class_id = CLASS_MAP[class_name]
                points = obj["polygon"]

                pts = np.array(
                    [[int(p["x"] * width), int(p["y"] * height)] for p in points],
                    dtype=np.int32
                )

                cv2.fillPoly(mask, [pts], class_id)

    mask_path = os.path.join(MASKS_DIR, external_id)
    cv2.imwrite(mask_path, mask)

    print(f"Saved mask: {mask_path}")

print("✅ All masks generated successfully.")
