import torch
from torch.utils.data import DataLoader
from dataset import SegDataset
from unet import UNet
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = SegDataset("dataset/train/images_aug", "dataset/train/masks_aug")
val_ds = SegDataset("dataset/val/images", "dataset/val/masks")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

model = UNet(n_classes=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "unet_model.pth")
print("✅ Training complete. Model saved.")
