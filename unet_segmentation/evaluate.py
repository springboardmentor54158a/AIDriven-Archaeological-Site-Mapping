import os
import cv2

IMG_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/predicted_masks"
OUT_DIR = "segmentation_dataset/overlay_results"

os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(MASK_DIR):
    img = cv2.imread(os.path.join(IMG_DIR, file))
    mask = cv2.imread(os.path.join(MASK_DIR, file), 0)

    if img is None or mask is None:
        continue

    overlay = img.copy()
    overlay[mask > 127] = [0, 255, 0]

    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imwrite(os.path.join(OUT_DIR, file), blended)

    print(f"Saved overlay: {file}")

print("Overlay completed")

