from PIL import Image
import numpy as np
import os

mask_dir = "dataset/val/masks"

for m in os.listdir(mask_dir):
    mask = Image.open(os.path.join(mask_dir, m))
    arr = np.array(mask)
    unique = np.unique(arr)
    print(m, "→ classes:", unique)
