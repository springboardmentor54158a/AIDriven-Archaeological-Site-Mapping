import os

img_dir = "images"
mask_dir = "masks"

imgs = os.listdir(img_dir)
masks = set(os.path.splitext(m)[0] for m in os.listdir(mask_dir))

kept = 0
removed = 0

for img in imgs:
    base = os.path.splitext(img)[0]
    if base not in masks:
        os.remove(os.path.join(img_dir, img))
        removed += 1
    else:
        kept += 1

print(f"Kept images: {kept}")
print(f"Removed images without masks: {removed}")
