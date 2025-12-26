import os
import cv2
import numpy as np

IMAGES_DIR = "dataset/train/images"
MASKS_DIR = "dataset/train/masks"

# Output folders (overwrite training folder with augmented data)
OUT_IMAGES = "dataset/train/images_aug"
OUT_MASKS = "dataset/train/masks_aug"

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

for fname in image_files:
    img = cv2.imread(os.path.join(IMAGES_DIR, fname))
    mask = cv2.imread(os.path.join(MASKS_DIR, fname), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    # Save original
    cv2.imwrite(os.path.join(OUT_IMAGES, fname), img)
    cv2.imwrite(os.path.join(OUT_MASKS, fname), mask)

    # Horizontal flip
    cv2.imwrite(os.path.join(OUT_IMAGES, f"hflip_{fname}"), cv2.flip(img, 1))
    cv2.imwrite(os.path.join(OUT_MASKS, f"hflip_{fname}"), cv2.flip(mask, 1))

    # Vertical flip
    cv2.imwrite(os.path.join(OUT_IMAGES, f"vflip_{fname}"), cv2.flip(img, 0))
    cv2.imwrite(os.path.join(OUT_MASKS, f"vflip_{fname}"), cv2.flip(mask, 0))

    # Rotation
    for angle in [-15, 15]:
        cv2.imwrite(os.path.join(OUT_IMAGES, f"rot{angle}_{fname}"), rotate(img, angle))
        cv2.imwrite(os.path.join(OUT_MASKS, f"rot{angle}_{fname}"), rotate(mask, angle))

print("✅ Data augmentation completed successfully.")
