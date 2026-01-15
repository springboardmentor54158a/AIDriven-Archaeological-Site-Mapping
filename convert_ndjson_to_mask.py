import json, os
from PIL import Image, ImageDraw

IMG_DIR = "images"
ANN_FILE = "labelbox_export2.ndjson"
MASK_DIR = "masks"

os.makedirs(MASK_DIR, exist_ok=True)

# 🔑 Labelbox label names → training IDs
LABEL_MAP = {
    "vegitation": 1,     # keep spelling exactly as in Labelbox
    "vegetation": 1,     # (added safety)
    "ruins": 2,
    "background": 0
}

with open(ANN_FILE) as f:
    for line in f:
        data = json.loads(line)

        img_name = data["data_row"]["external_id"]
        img_path = os.path.join(IMG_DIR, img_name)

        # Handle .jpg / .png mismatch
        if not os.path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg") if img_path.endswith(".png") else img_path.replace(".jpg", ".png")

        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_name}")
            continue

        img = Image.open(img_path)
        w, h = img.size

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for project in data["projects"].values():
            for label in project["labels"]:
                for obj in label["annotations"]["objects"]:
                    label_name = obj["value"].lower()

                    if label_name not in LABEL_MAP:
                        print(f"⚠️ Unknown label: {label_name}")
                        continue

                    cls = LABEL_MAP[label_name]
                    poly = [(p["x"], p["y"]) for p in obj["polygon"]]
                    draw.polygon(poly, fill=cls)

        out_name = os.path.basename(img_path).replace(".jpg", ".png")
        mask.save(os.path.join(MASK_DIR, out_name))
        print(f"✅ Mask created for {out_name}")

print("🎉 All masks generated successfully")
