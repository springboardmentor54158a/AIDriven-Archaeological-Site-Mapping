
import numpy as np

def compute_iou_dice(pred, gt, cls):
    pred = pred == cls
    gt = gt == cls
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 0
    dice = 2*inter / (pred.sum()+gt.sum()) if pred.sum()+gt.sum() else 0
    return iou, dice
