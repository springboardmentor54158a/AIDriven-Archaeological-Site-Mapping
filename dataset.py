import os, torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, filelist, size=(512, 512)):
        self.files = open(filelist).read().splitlines()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        base = os.path.splitext(name)[0]
        mask = Image.open(os.path.join(self.mask_dir, base + ".png"))

        # Resize
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        img = TF.to_tensor(img)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return img, mask

    def __len__(self):
        return len(self.files)
