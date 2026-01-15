# Interactive Dashboard - Ruins & Vegetation Analysis

## Overview

This interactive dashboard integrates three AI models for comprehensive terrain analysis:
1. **Segmentation** (DeepLabV3) - Pixel-level classification
2. **Detection** (YOLOv5) - Object detection and classification  
3. **Erosion Prediction** (XGBoost) - Terrain erosion risk assessment

## Features

✅ **Single Image Analysis**
- Upload an image or select from dataset
- View unified visualization with all three models
- Detailed metrics and statistics

✅ **Batch Processing**
- Process multiple images at once
- Export results to CSV
- Summary statistics

✅ **Interactive Configuration**
- Adjustable model paths
- Configurable detection thresholds
- Device selection (CPU/CUDA)

## Installation

```bash
# Install requirements
pip install -r requirements_dashboard.txt

# Or install individually
pip install streamlit torch torchvision numpy pillow opencv-python pandas matplotlib scikit-learn xgboost joblib
```

## Running the Dashboard

### Method 1: Using the launcher script
```bash
python run_dashboard.py
```

### Method 2: Direct Streamlit command
```bash
streamlit run dashboard.py
```

### Method 3: With custom port
```bash
streamlit run dashboard.py --server.port 8501
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Usage

### 1. Single Image Analysis

1. Go to **"Upload Image"** tab
2. Either:
   - Upload an image file (JPG, PNG, JPEG)
   - Or check "Use existing image" and select from dataset
3. Configure model paths in sidebar (if needed)
4. View results:
   - Original image
   - Segmentation mask
   - Object detections
   - Erosion prediction

### 2. Batch Analysis

1. Go to **"Batch Analysis"** tab
2. Upload multiple images
3. Wait for processing to complete
4. View summary table
5. Download results as CSV

### 3. Configuration

**Sidebar Settings:**
- **Segmentation Model Path**: Optional path to trained DeepLab model
- **Detection Model Path**: Path to YOLOv5 weights (default: `runs/train/artifacts_quick2/weights/best.pt`)
- **Erosion Model Path**: Path to XGBoost model (default: `erosion_xgboost_regression.pkl`)
- **Confidence Threshold**: Detection confidence (0.0-1.0)
- **IoU Threshold**: Non-maximum suppression threshold
- **Device**: CPU or CUDA

## Model Integration Details

### Segmentation Model
- **Model**: DeepLabV3-ResNet50
- **Classes**: Background (0), Vegetation (1), Ruins (2)
- **Output**: Pixel-level segmentation mask
- **Metrics**: Coverage percentages

### Detection Model
- **Model**: YOLOv5s
- **Classes**: Vegetation (0), Ruins (1)
- **Output**: Bounding boxes with confidence scores
- **Metrics**: Object count, detection confidence

### Erosion Prediction Model
- **Model**: XGBoost Regressor
- **Features**: 21 terrain features (vegetation index, slope, texture, etc.)
- **Output**: Erosion probability (0-1)
- **Metrics**: Risk level (Stable/Moderate/High)

## Output Visualization

The dashboard creates a unified 2x2 grid visualization:
- **Top Left**: Original image
- **Top Right**: Segmentation mask (colored by class)
- **Bottom Left**: Object detections (bounding boxes)
- **Bottom Right**: Erosion prediction with risk level

## File Structure

```
.
├── dashboard.py              # Main Streamlit dashboard
├── inference_models.py      # Model loading and inference functions
├── run_dashboard.py         # Quick launcher script
├── requirements_dashboard.txt # Dependencies
└── DASHBOARD_README.md      # This file
```

## Troubleshooting

### Models Not Loading
- Check model paths in sidebar
- Ensure model files exist
- Check file permissions

### CUDA Not Available
- Dashboard will automatically fall back to CPU
- Check CUDA installation if needed

### Import Errors
- Install all requirements: `pip install -r requirements_dashboard.txt`
- Ensure all model dependencies are installed

### YOLOv5 Loading Issues
- Ensure `yolov5` directory exists (cloned from GitHub)
- Check YOLOv5 weights file path

## Performance Tips

1. **Use CPU for small images** - Faster startup
2. **Use CUDA for batch processing** - Faster inference
3. **Reduce image size** - Faster processing
4. **Adjust confidence threshold** - Balance accuracy vs speed

## Example Workflow

1. **Start Dashboard**: `streamlit run dashboard.py`
2. **Upload Image**: Select a ruins/vegetation image
3. **View Results**: See all three model outputs
4. **Analyze Metrics**: Check coverage, detections, erosion risk
5. **Export**: Download results if needed

## Integration with Other Components

The dashboard uses:
- `terrain_features.py` - For erosion feature extraction
- `erosion_labeled_data.csv` - For feature column names
- Trained model weights from previous milestones

## Next Steps

- Add real-time webcam support
- Integrate with GIS systems
- Add export to GeoJSON
- Create API endpoints
- Add model comparison views

## Support

For issues or questions:
1. Check model paths are correct
2. Verify all dependencies are installed
3. Check console output for error messages
4. Ensure images are in correct format (RGB)

---

**Dashboard Version**: 1.0  
**Last Updated**: Milestone 4, Week 7

