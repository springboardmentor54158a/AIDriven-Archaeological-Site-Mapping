import os, random

IMG_DIR = "images"
MASK_DIR = "masks"

# get base names (without extension)
imgs = {os.path.splitext(f)[0]: f for f in os.listdir(IMG_DIR)}
masks = {os.path.splitext(f)[0]: f for f in os.listdir(MASK_DIR)}

# intersection
common = sorted(set(imgs.keys()) & set(masks.keys()))

print("Total images:", len(imgs))
print("Total masks:", len(masks))
print("Matched pairs:", len(common))

# recreate clean train / val lists
random.shuffle(common)
split = int(0.8 * len(common))

with open("train.txt", "w") as f:
    for b in common[:split]:
        f.write(imgs[b] + "\n")

with open("val.txt", "w") as f:
    for b in common[split:]:
        f.write(imgs[b] + "\n")

print("Train images:", split)
print("Val images:", len(common) - split)
