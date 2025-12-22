import cv2
import os
import numpy as np

GT_DIR = r"dataset\val\masks"
PRED_DIR = r"results\predictions"

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-7)

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2 * intersection) / (y_true.sum() + y_pred.sum() + 1e-7)

ious, dices = [], []

for file in os.listdir(GT_DIR):
    gt = cv2.imread(os.path.join(GT_DIR, file), 0) > 127
    pred = cv2.imread(os.path.join(PRED_DIR, file), 0) > 127

    ious.append(iou_score(gt, pred))
    dices.append(dice_score(gt, pred))

print("Mean IoU :", np.mean(ious))
print("Mean Dice:", np.mean(dices))
