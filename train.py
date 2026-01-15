import torch, numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from dataset import SegDataset

NUM_CLASSES = 3
EPOCHS = 10
BATCH = 2
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def iou(pred, target, cls):
    inter = ((pred == cls) & (target == cls)).sum().item()
    union = ((pred == cls) | (target == cls)).sum().item()
    return inter / (union + 1e-6)

def dice(pred, target, cls):
    inter = ((pred == cls) & (target == cls)).sum().item()
    return 2 * inter / ((pred == cls).sum().item() + (target == cls).sum().item() + 1e-6)

train_ds = SegDataset("images", "masks", "train.txt")
val_ds = SegDataset("images", "masks", "val.txt")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch} ---")

    model.train()
    total_loss = 0

    for step, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        out = model(imgs)["out"]
        loss = loss_fn(out, masks)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        print(f"Step {step} | Loss {loss.item():.4f}")

    model.eval()
    ious, dices = [], []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)["out"].argmax(1)
            for c in [0, 1, 2]:
                ious.append(iou(preds, masks, c))
                dices.append(dice(preds, masks, c))

    print(
        f"Epoch {epoch} DONE | "
        f"Loss={total_loss/len(train_loader):.4f} | "
        f"mIoU={np.mean(ious):.4f} | "
        f"Dice={np.mean(dices):.4f}"
    )

print("Training completed ✅")
