import os
import random
import shutil

IMAGES_DIR = "images"
MASKS_DIR = "masks"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

# Only images that have masks
image_files = [f for f in os.listdir(MASKS_DIR) if f.endswith(".png")]
random.shuffle(image_files)

total = len(image_files)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

for split, files in splits.items():
    for f in files:
        shutil.copy(os.path.join(IMAGES_DIR, f),
                    os.path.join(OUTPUT_DIR, split, "images", f))
        shutil.copy(os.path.join(MASKS_DIR, f),
                    os.path.join(OUTPUT_DIR, split, "masks", f))

print("✅ Dataset split completed successfully.")
