import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from unet import UNet  # your unet.py file

# -----------------------------
# PATHS
# -----------------------------
VAL_IMG_DIR = r"D:\Infosys Project\segmentation\dataset\val\images"
MODEL_PATH = r"D:\Infosys Project\segmentation\unet_model.pth"
OUTPUT_DIR = r"D:\Infosys Project\segmentation\pred_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (256, 256)  # if training used different size, change it
DEVICE = "cpu"

print("✅ VAL_IMG_DIR:", VAL_IMG_DIR)
print("✅ MODEL_PATH:", MODEL_PATH)
print("✅ OUTPUT_DIR:", OUTPUT_DIR)

# -----------------------------
# CHECK IMAGES
# -----------------------------
if not os.path.exists(VAL_IMG_DIR):
    print("❌ ERROR: val images folder not found!")
    exit()

image_files = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"🔍 Total validation images found: {len(image_files)}")

if len(image_files) == 0:
    print("❌ ERROR: No images found inside validation folder!")
    exit()

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file not found!")
    exit()

model = UNet(n_classes=4)  # Ruins=0, Vegetation=1, Terrain=2 (as per your project)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully!")

# -----------------------------
# PREDICT + SAVE
# -----------------------------
saved_count = 0

for file in tqdm(image_files):
    img_path = os.path.join(VAL_IMG_DIR, file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {file}")
        continue

    # Convert BGR -> RGB and resize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_tensor)  # [1, C, H, W]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()  # [H, W]

    # Create visible grayscale mask
    mask_vis = (pred * 127).astype(np.uint8)

    # Save predicted mask as PNG always
    out_name = os.path.splitext(file)[0] + "_pred.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    ok = cv2.imwrite(out_path, mask_vis)

    if ok:
        saved_count += 1
    else:
        print(f"❌ Failed to save mask: {out_path}")

print(f"\n✅ DONE! Total predicted masks saved: {saved_count}")
print("📁 Output folder:", OUTPUT_DIR)
