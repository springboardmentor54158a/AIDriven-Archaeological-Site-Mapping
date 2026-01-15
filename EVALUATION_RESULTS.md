# YOLOv5 Artifact Detection - Real Evaluation Results

## Training Configuration
- **Model**: YOLOv5s (small)
- **Epochs**: 3 (reduced for speed)
- **Batch Size**: 2
- **Image Size**: 416x416
- **Device**: CPU
- **Training Images**: 16
- **Validation Images**: 5

## Real Evaluation Metrics

### Overall Performance (All Classes Combined)

| Metric | Value |
|--------|-------|
| **Precision** | 0.0028 (0.28%) |
| **Recall** | 0.6000 (60.00%) |
| **mAP@0.5** | 0.0211 (2.11%) |
| **mAP@0.5:0.95** | 0.00808 (0.808%) |

### Class-wise Performance

#### 1. Vegetation (Class 0)
| Metric | Value |
|--------|-------|
| **Precision** | 0.00436 (0.436%) |
| **Recall** | 1.0000 (100%) |
| **mAP@0.5** | 0.0376 (3.76%) |
| **mAP@0.5:0.95** | 0.0139 (1.39%) |
| **Instances** | 3 (in validation set) |

#### 2. Ruins (Class 1)
| Metric | Value |
|--------|-------|
| **Precision** | 0.00123 (0.123%) |
| **Recall** | 0.2000 (20%) |
| **mAP@0.5** | 0.00456 (0.456%) |
| **mAP@0.5:0.95** | 0.00228 (0.228%) |
| **Instances** | 5 (in validation set) |

## Interpretation

### What These Metrics Mean:

1. **Precision**: Out of all detections made, how many were correct?
   - Very low precision indicates many false positives
   - Model is detecting objects but with low confidence/accuracy

2. **Recall**: Out of all ground truth objects, how many were detected?
   - **Vegetation**: 100% recall means all vegetation instances were detected
   - **Ruins**: 20% recall means only 1 out of 5 ruins were detected

3. **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
   - Measures detection accuracy when boxes overlap by at least 50%
   - Vegetation performs better (3.76%) than ruins (0.456%)

4. **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95
   - More strict metric requiring better box localization
   - Overall: 0.808%

### Why Metrics Are Low:

- **Only 3 epochs**: Model needs more training time to learn properly
- **Small dataset**: 16 training images is very small for deep learning
- **CPU training**: Slower convergence compared to GPU
- **Early training stage**: Model is still learning basic features

### Recommendations for Improvement:

1. **Train for more epochs**: 20-50 epochs minimum
2. **Increase dataset size**: More training images (100+ recommended)
3. **Use GPU**: Faster training allows more experimentation
4. **Data augmentation**: Already enabled in YOLOv5
5. **Fine-tune hyperparameters**: Learning rate, batch size, etc.

## Files Generated

- **Model weights**: `runs/train/artifacts_quick2/weights/best.pt`
- **Training plots**: `runs/train/artifacts_quick2/`
- **Evaluation results**: `evaluation_results.json`
- **Validation plots**: `runs/val/eval/` (confusion matrix, PR curves, F1 curves)

## Next Steps

To improve results, run:
```bash
python train_yolo.py --data dataset_yolo.yaml --model yolov5s --epochs 20 --batch-size 4 --img-size 640 --device cpu --name artifacts_full
```

Then evaluate:
```bash
python evaluate_yolo.py --weights runs/train/artifacts_full/weights/best.pt --data dataset_yolo.yaml
```

