# Archaeological Analysis Dashboard

## 🏛️ Overview

An interactive Streamlit dashboard that integrates three powerful AI models for comprehensive archaeological site analysis:

1. **YOLO Object Detection** - Identify artifacts, ruins, and structures
2. **U-Net Segmentation** - Pixel-level segmentation of ruins and vegetation
3. **Terrain Erosion Prediction** - Assess erosion risk using ML

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install streamlit plotly opencv-python ultralytics torch torchvision pandas matplotlib seaborn scikit-learn joblib pillow segmentation-models-pytorch
```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## 📊 How It Works

**Step 1: Upload an Image**
Simply upload a single archaeological site image (JPG/PNG).

**Step 2: Automatic Analysis**
The system automatically runs all three models in parallel:
- **YOLO** detects and bounds objects.
- **U-Net** segments the image into ruins (red) and vegetation (green).
- **ML Model** predicts the erosion risk based on vegetation and slope features.

**Step 3: View Results**
- **Summary**: A high-level overview of detections, coverage, and risk.
- **Detailed Tabs**: Click through tabs to see rich visualizations for each model (annotated images, distribution charts, and risk gauges).

## 📁 Required Files

The dashboard expects the following files in the same directory:

- `archaeological_yolo_best.pt` - Trained YOLO model
- `unet_archaeology.pth` - Trained U-Net model
- `erosion_model.pkl` - Trained erosion prediction model

## 🔧 Configuration

### Sidebar Settings
- **Detection Confidence**: Adjust YOLO detection threshold (0.0 - 1.0)
- **Model Status**: Real-time status of available models

## 🐛 Troubleshooting

### RGBA Image Error
The dashboard automatically converts RGBA (4-channel) images to RGB (3-channel) to prevent model errors.

### Missing Models
If models are not found, the dashboard will display warnings in the sidebar and provide helpful messages.

## 🔬 Technical Details

### Models
- **YOLO**: YOLOv8 for object detection (3 classes: artifact, ruins, structure)
- **U-Net**: ResNet34 encoder for semantic segmentation (3 classes: background, ruins, vegetation)
- **RandomForest**: 200 estimators for erosion prediction (2 classes: stable, erosion-prone)

## 📝 License

This dashboard is part of the Archaeological Analysis project.
