import cv2
import os
import numpy as np

IMG_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"
IMG_SIZE = 256

def preprocess():
    for folder in [IMG_DIR, MASK_DIR]:
        for file in os.listdir(folder):
            if file.endswith(".png"):
                path = os.path.join(folder, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if folder == MASK_DIR else cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(path, img)

    print(" Preprocessing completed (256x256 resize)")

if __name__ == "__main__":
    preprocess()

