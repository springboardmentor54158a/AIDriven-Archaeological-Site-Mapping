import torch
import cv2
import os
import numpy as np
from src.model import get_unet

# -------- PATHS --------
IMG_DIR = r"dataset\val\images"
OUT_DIR = r"results\predictions"
MODEL_PATH = r"results\models\unet_model.pth"

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MODEL --------
model = get_unet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------- INFERENCE LOOP --------
for file in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, file)

    # ---- Read image ----
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0

    # ---- To tensor ----
    tensor = torch.tensor(image, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # ---- Model prediction ----
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)

    # ---- Thresholding ----
    pred = (pred > 0.4).float()

    # ---- Convert to numpy mask ----
    pred_mask = pred.squeeze().cpu().numpy().astype("uint8") * 255

    # ---- Morphological cleaning ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

    # ---- Save mask ----
    mask_path = os.path.join(OUT_DIR, file)
    cv2.imwrite(mask_path, pred_mask)

    # ---- Overlay on original image (for visualization) ----
    overlay = (image * 255).astype("uint8").copy()
    overlay[pred_mask > 0] = [255, 0, 0]  # red mask

    blended = cv2.addWeighted(
        (image * 255).astype("uint8"),
        0.6,
        overlay,
        0.4,
        0
    )

    overlay_path = os.path.join(
        OUT_DIR, file.replace(".png", "_overlay.png")
    )
    cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    print(f"✅ Predicted: {file}")

print("\n🎉 Inference completed successfully")
