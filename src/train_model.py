"""
Final Training Pipeline Script (Placeholder)

Note:
- Model training was already performed using YOLO / UNet experiments.
- This script exists to represent the final pipeline stage in the project.
- Trained models are stored inside the 'models/' directory.
"""

import os

MODEL_DIR = "models"

def main():
    if not os.path.exists(MODEL_DIR):
        print("Models directory not found.")
        return

    models = os.listdir(MODEL_DIR)

    if len(models) == 0:
        print("No trained models found.")
    else:
        print("Available trained models:")
        for model in models:
            print("-", model)

    print("\nTraining phase already completed using image-based datasets.")

if __name__ == "__main__":
    main()
