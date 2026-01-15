# Pillars and Sculptures Detection Setup Guide

This guide explains how to set up YOLOv5 to specifically detect pillars and sculptures as separate classes.

## Overview

The current model has 2 classes:
- Class 0: Vegetation
- Class 1: Ruins

We're adding 2 new classes:
- Class 2: Pillars
- Class 3: Sculptures

## Step 1: Prepare Your Dataset

### Option A: Convert Existing Labels (Quick Start)

If you already have labeled data with "ruins" class, you can convert it:

```bash
python label_pillars_sculptures.py --labels-dir yolo_dataset/train/labels
```

This will:
- Create a backup of your original labels
- Create new label files ready for 4-class labeling
- Create a labeling guide

### Option B: Label from Scratch

1. Use a labeling tool like [LabelImg](https://github.com/tzutalin/labelImg) or [Roboflow](https://roboflow.com)
2. Label images with 4 classes:
   - **Vegetation** (class 0): Trees, bushes, grass
   - **Ruins** (class 1): General ruins, broken structures
   - **Pillars** (class 2): Vertical columns, posts, structural supports
   - **Sculptures** (class 3): Statues, carvings, decorative elements

## Step 2: Update Dataset Configuration

The new dataset config file `dataset_yolo_pillars_sculptures.yaml` is already created with 4 classes.

Make sure the paths are correct:
```yaml
path: yolo_dataset  # or your dataset path
train: train/images
val: val/images

names:
  0: vegetation
  1: ruins
  2: pillars
  3: sculptures

nc: 4
```

## Step 3: Train the Model

Train a new model specifically for pillars and sculptures:

```bash
python train_pillars_sculptures.py \
    --data dataset_yolo_pillars_sculptures.yaml \
    --model yolov5s \
    --epochs 50 \
    --batch-size 8 \
    --img-size 640 \
    --device cpu
```

### Training Tips:

- **More epochs**: Use 50-100 epochs for better results
- **Larger batch size**: If you have GPU, increase batch-size (16, 32)
- **Image size**: 640 is good, 832 or 1280 for better small object detection
- **GPU**: Use `--device cuda:0` if you have GPU (much faster)

### Expected Training Time:
- CPU: ~2-4 hours for 50 epochs
- GPU: ~30-60 minutes for 50 epochs

## Step 4: Use the Trained Model

Once training is complete, update the dashboard to use the new model:

1. In the dashboard sidebar, set "Detection Model Path" to:
   ```
   runs/train/pillars_sculptures/weights/best.pt
   ```

2. The dashboard will automatically detect the 4 classes and show:
   - Vegetation (green boxes)
   - Ruins (red boxes)
   - Pillars (orange boxes)
   - Sculptures (magenta boxes)

## Step 5: Improve Detection

### For Better Pillar/Sculpture Detection:

1. **Lower confidence threshold**: Try 0.05-0.1 in the dashboard sidebar
2. **Larger inference size**: Use 832 or 1280 for small objects
3. **More training data**: Add more examples of pillars and sculptures
4. **Data augmentation**: The training script includes augmentation automatically

### Labeling Tips:

- **Pillars**: Vertical columns, posts, structural supports (even if broken)
- **Sculptures**: Statues, carvings, decorative/artistic elements
- **When in doubt**: Use "ruins" (class 1) - better to have fewer false positives

## Troubleshooting

### Model not detecting pillars/sculptures?

1. Check if you're using the correct model path (4-class model)
2. Lower the confidence threshold (try 0.05)
3. Increase inference size (try 832 or 1280)
4. Verify your labels are correct (check label files)

### Training errors?

1. Make sure dataset paths in YAML are correct
2. Check that label files exist for all images
3. Verify class IDs are 0, 1, 2, 3 (not higher)
4. Ensure you have enough images (recommended: 50+ per class)

### Poor detection performance?

1. Train for more epochs (100+)
2. Add more training data, especially for pillars/sculptures
3. Use data augmentation
4. Try a larger model (yolov5m or yolov5l)

## Quick Start Commands

```bash
# 1. Convert existing labels
python label_pillars_sculptures.py

# 2. Review and update labels manually (use LabelImg or similar)

# 3. Train the model
python train_pillars_sculptures.py --epochs 50 --device cpu

# 4. Run dashboard with new model
python run_dashboard.py
# Then set model path to: runs/train/pillars_sculptures/weights/best.pt
```

## Model Performance Expectations

With proper training (50+ epochs, good dataset):
- **Pillars**: Should detect vertical columns and posts
- **Sculptures**: Should detect statues and carvings
- **Confidence**: May need lower threshold (0.05-0.1) for small objects

## Next Steps

1. Collect more images with pillars and sculptures
2. Label them carefully using the guide
3. Retrain with more data
4. Fine-tune confidence thresholds in the dashboard
