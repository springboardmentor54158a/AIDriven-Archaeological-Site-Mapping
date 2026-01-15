import os
import numpy as np
from PIL import Image

classes = set()

for f in os.listdir("masks"):
    mask = np.array(Image.open(os.path.join("masks", f)))
    classes |= set(np.unique(mask))

print("Classes found in dataset:", classes)
