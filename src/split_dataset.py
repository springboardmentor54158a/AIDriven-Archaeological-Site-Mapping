import os
import shutil
import random

# SOURCE (already processed data)
IMG_DIR = r"dataset\processed\images"
MASK_DIR = r"dataset\processed\masks"

# DESTINATION
TRAIN_IMG = r"dataset\train\images"
TRAIN_MASK = r"dataset\train\masks"
VAL_IMG = r"dataset\val\images"
VAL_MASK = r"dataset\val\masks"

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_MASK, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_MASK, exist_ok=True)

files = os.listdir(IMG_DIR)
random.shuffle(files)

split = int(0.8 * len(files))  # 80% train
train_files = files[:split]
val_files = files[split:]

for f in train_files:
    shutil.copy(os.path.join(IMG_DIR, f), os.path.join(TRAIN_IMG, f))
    shutil.copy(os.path.join(MASK_DIR, f), os.path.join(TRAIN_MASK, f))

for f in val_files:
    shutil.copy(os.path.join(IMG_DIR, f), os.path.join(VAL_IMG, f))
    shutil.copy(os.path.join(MASK_DIR, f), os.path.join(VAL_MASK, f))

print("✅ Dataset split completed")
print("Train images:", len(train_files))
print("Val images:", len(val_files))
