import os
import cv2
import numpy as np
import tensorflow as tf

IMG_DIR = "segmentation_dataset/images"
OUT_DIR = "segmentation_dataset/predicted_masks"
IMG_SIZE = 256

os.makedirs(OUT_DIR, exist_ok=True)

model = tf.keras.models.load_model("unet_model.h5")
print("Model loaded")

for file in os.listdir(IMG_DIR):
    if not file.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(IMG_DIR, file))
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_norm)[0]
    pred = (pred > 0.5).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(OUT_DIR, file), pred)
    print(f" Predicted {file}")

print(" Prediction completed")

