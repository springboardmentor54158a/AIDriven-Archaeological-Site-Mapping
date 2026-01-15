# Quick Start: Pillars & Sculptures Detection

## What's Been Set Up

✅ **4-Class YOLO Configuration** - Ready for pillars and sculptures detection
✅ **Training Script** - `train_pillars_sculptures.py`
✅ **Labeling Helper** - `label_pillars_sculptures.py`
✅ **Updated Dashboard** - Supports 4-class detection with color coding

## Quick Start (3 Steps)

### Step 1: Prepare Your Labels

If you have existing labels, convert them:
```bash
python label_pillars_sculptures.py
```

Then manually update labels using LabelImg or similar tool:
- Change some "ruins" (class 1) → "pillars" (class 2)
- Change some "ruins" (class 1) → "sculptures" (class 3)

### Step 2: Train the Model

```bash
python train_pillars_sculptures.py --epochs 50 --device cpu
```

For GPU (much faster):
```bash
python train_pillars_sculptures.py --epochs 50 --device cuda:0 --batch-size 16
```

### Step 3: Use in Dashboard

1. Run dashboard: `python run_dashboard.py`
2. In sidebar, set model path to: `runs/train/pillars_sculptures/weights/best.pt`
3. Lower confidence to 0.05-0.1 for better detection
4. Increase inference size to 832 or 1280

## Color Coding in Dashboard

- 🟢 **Green**: Vegetation
- 🔴 **Red**: Ruins (general)
- 🟠 **Orange**: Pillars
- 🟣 **Magenta**: Sculptures

## Current Model Status

The current model (`artifacts_quick2`) has only 2 classes. For pillars/sculptures:
- **Option 1**: Train new 4-class model (recommended)
- **Option 2**: Use current model with lower confidence (may detect some as "ruins")

## Need Help?

See `PILLARS_SCULPTURES_SETUP.md` for detailed instructions.
