import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SegDataset
from unet import UNet
from tqdm import tqdm

# ---------- SETTINGS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
MODEL_PATH = "unet_model.pth"

# ---------- LOAD DATA ----------
val_dataset = SegDataset("dataset/train/images", "dataset/train/masks", size=512)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ---------- LOAD MODEL ----------
model = UNet(n_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- METRIC FUNCTIONS ----------
def iou_score(pred, target, cls):
    pred = (pred == cls)
    target = (target == cls)
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return np.nan
    return intersection / union

def dice_score(pred, target, cls):
    pred = (pred == cls)
    target = (target == cls)
    intersection = (pred & target).sum()
    total = pred.sum() + target.sum()
    if total == 0:
        return np.nan
    return (2 * intersection) / total

# ---------- EVALUATION ----------
iou_scores = {c: [] for c in range(1, NUM_CLASSES)}
dice_scores = {c: [] for c in range(1, NUM_CLASSES)}

with torch.no_grad():
    for imgs, masks in tqdm(val_loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        preds = preds.cpu().numpy()[0]
        masks = masks.cpu().numpy()[0]

        for cls in range(1, NUM_CLASSES):
            if cls not in masks:
                 continue
            iou = iou_score(preds, masks, cls)
            dice = dice_score(preds, masks, cls)
            iou_scores[cls].append(iou)
            dice_scores[cls].append(dice)


# ---------- RESULTS ----------
print("\n===== VALIDATION RESULTS =====")
class_names = {1: "Ruins", 2: "Vegetation", 3: "Terrain"}

for cls in iou_scores:
    mean_iou = np.mean(iou_scores[cls]) * 100
    mean_dice = np.mean(dice_scores[cls]) * 100
    print(f"{class_names[cls]} → IoU: {mean_iou:.2f}% | Dice: {mean_dice:.2f}%")

overall_iou = np.mean([np.mean(iou_scores[c]) for c in iou_scores]) * 100
overall_dice = np.mean([np.mean(dice_scores[c]) for c in dice_scores]) * 100

print(f"\nOverall Mean IoU: {overall_iou:.2f}%")
print(f"Overall Mean Dice: {overall_dice:.2f}%")
