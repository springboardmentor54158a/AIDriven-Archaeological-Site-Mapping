import torch
from torch.utils.data import DataLoader
from src.dataset_loader import SegmentationDataset
from src.model import get_unet
import os
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

# -------- PATHS --------
# -------- PATHS --------
TRAIN_IMG = r"dataset\train\images"
TRAIN_MASK = r"dataset\train\masks"

VAL_IMG = r"dataset\val\images"
VAL_MASK = r"dataset\val\masks"


# -------- TRAINING CONFIG --------
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- DATA --------
train_dataset = SegmentationDataset(TRAIN_IMG, TRAIN_MASK)
val_dataset = SegmentationDataset(VAL_IMG, VAL_MASK)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------- MODEL --------
model = get_unet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# -------- TRAIN LOOP --------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# -------- SAVE MODEL --------
os.makedirs("results/models", exist_ok=True)
torch.save(model.state_dict(), "results/models/unet_model.pth")

print("\n✅ Training complete. Model saved.")
