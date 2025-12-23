import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
class VegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir, img_name.replace(".jpg", "_mask.png")
        )

        # Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image / 255.0

        # Mask (color → class)
        mask_color = cv2.imread(mask_path)
        mask_color = cv2.resize(mask_color, (512, 512))

        mask = np.zeros((512, 512), dtype=np.uint8)

        green = (mask_color[:, :, 1] > 200) & (mask_color[:, :, 0] < 50)
        mask[green] = 1   # vegetation

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return image, mask
train_dataset = VegDataset(
    "/content/dataset/images",
    "/content/dataset/masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2   # background + vegetation
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(train_loader):.4f}")
def dice_score(pred, gt):
    pred = (pred == 1).astype(np.uint8)
    gt = (gt == 1).astype(np.uint8)
    intersection = (pred & gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum() + 1e-7)

def iou_score(pred, gt):
    pred = (pred == 1).astype(np.uint8)
    gt = (gt == 1).astype(np.uint8)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / (union + 1e-7)
model.eval()

image, gt_mask = train_dataset[0]
image_tensor = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)

pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
gt_mask = gt_mask.numpy()

print("Dice:", round(dice_score(pred_mask, gt_mask), 4))
print("IoU :", round(iou_score(pred_mask, gt_mask), 4))
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Input Image")
plt.imshow(image.permute(1,2,0))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(gt_mask, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("U-Net Prediction")
plt.imshow(pred_mask, cmap="jet")
plt.axis("off")

plt.show()
