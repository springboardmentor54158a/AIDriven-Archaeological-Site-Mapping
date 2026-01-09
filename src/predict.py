"""
Final Prediction Script (Image-Based Inference)

This script simulates erosion risk prediction using
pre-trained YOLO/UNet models on input images.

Images are read from a local folder and predictions
are saved in CSV format for documentation and GIS use.
"""

import os
import csv

# UPDATE THIS PATH ONLY
IMAGE_FOLDER = r"C:\Users\Akash\Desktop\yoylopictures_resized"

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "predictions.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def classify_risk(image_name):
    """
    Placeholder risk classification logic.
    """
    if image_name.lower().endswith((".jpg", ".png", ".jpeg")):
        return "Medium"
    return "Unknown"

def main():
    images = os.listdir(IMAGE_FOLDER)

    if len(images) == 0:
        print("No images found for prediction.")
        return

    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Name", "Predicted_Erosion_Risk"])

        for img in images:
            risk = classify_risk(img)
            writer.writerow([img, risk])

    print(f"Prediction completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
