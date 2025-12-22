# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# --- 1. Load and Prepare Your Image ---
image_path = "C:\\Users\\Akash\\Desktop\\input_image.png"
input_image = Image.open(image_path).convert('RGB') # Ensure it's RGB

# Resize and convert to tensor (U-Net often expects specific sizes)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a square dimension
    transforms.ToTensor(),           # Convert to PyTorch tensor
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension -> shape [1, 3, 256, 256]

# --- 2. Load a Pre-trained U-Net Model ---
# For a real project, you would load a .pth file from training.
# For this demo, we'll create a simple U-Net structure and output a dummy mask.
# This shows you the pipeline. In practice, you would load `model.load_state_dict(torch.load('model.pth'))`
print("[Info] In a full project, you would load your trained U-Net model here.")

# --- 3. Generate a Prediction (Dummy Example) ---
# Since we don't have a real trained model for ruins yet,
# we'll create a synthetic "mask" for demonstration.
# A real model output would be a tensor of probabilities.
with torch.no_grad():
    # This simulates a model's prediction: random values between 0-1
    simulated_output = torch.rand(1, 1, 256, 256)

# --- 4. Process and Visualize the Output ---
# Convert probability to a binary mask (threshold = 0.5)
mask_numpy = simulated_output.squeeze().numpy()  # Remove batch & channel dims
binary_mask = (mask_numpy > 0.5).astype(np.uint8) * 255  # Scale to 0 or 255

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(input_image)
axes[0].set_title("Your Input Image")
axes[0].axis('off')

axes[1].imshow(mask_numpy, cmap='gray')
axes[1].set_title("Model's Raw Output")
axes[1].axis('off')

axes[2].imshow(binary_mask, cmap='gray')
axes[2].set_title("Binary Mask (Threshold > 0.5)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("Demo complete! With a real trained model, the mask would highlight ruins.")
