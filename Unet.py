import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

image = cv2.imread("/content/veg10.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))
image = image / 255.0

mask_color = cv2.imread("/content/veg10mask.png")   
mask_color = cv2.resize(mask_color, (512, 512))

mask = np.zeros((512, 512), dtype=np.uint8)

green = (mask_color[:,:,1] > 200) & (mask_color[:,:,0] < 50)
mask[green] = 1

black = (mask_color[:,:,0] < 30) & (mask_color[:,:,1] < 30) & (mask_color[:,:,2] < 30)
mask[black] = 2

image_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float()

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)

model.eval()

with torch.no_grad():
    output = model(image_tensor)

prediction = torch.argmax(output, dim=1).squeeze().numpy()

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Input Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ground Truth Mask")
plt.imshow(mask, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("U-Net Prediction (Demo)")
plt.imshow(prediction, cmap="jet")
plt.axis("off")

plt.show()
