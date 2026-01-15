# AI-Driven Archaeological Site Mapping

An integrated AI system for analyzing archaeological sites using computer vision and machine learning. This project combines segmentation, object detection, and erosion prediction models to provide comprehensive terrain analysis.

## Features

### 🎯 Three Integrated Models

1. **Segmentation (DeepLabV3)**
   - Pixel-level classification of vegetation and ruins
   - Coverage percentage calculation
   - Background, Vegetation, and Ruins detection

2. **Object Detection (YOLOv5)**
   - Detects vegetation and ruins objects
   - Supports pillars and sculptures detection (4-class model)
   - Bounding box visualization with confidence scores

3. **Erosion Prediction (XGBoost)**
   - Terrain erosion probability prediction
   - Risk level assessment (Stable/Moderate/High)
   - Feature-based analysis

### 📊 Interactive Dashboard

- **Single Image Analysis**: Upload and analyze individual images
- **Batch Processing**: Process multiple images at once
- **Unified Visualization**: Combined view of all model outputs
- **Detailed Metrics**: Coverage percentages, detection counts, erosion risk

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/springboardmentor54158a/AIDriven-Archaeological-Site-Mapping.git
cd AIDriven-Archaeological-Site-Mapping
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements_dashboard.txt
```

### Running the Dashboard

```bash
python run_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## Training Models

### YOLOv5 Detection Model

Train a 2-class model (vegetation, ruins):
```bash
python train_yolo.py --data dataset_yolo.yaml --epochs 30 --batch-size 4 --img-size 640
```

Train a 4-class model (vegetation, ruins, pillars, sculptures):
```bash
python train_pillars_sculptures.py --epochs 50 --batch-size 8 --img-size 640
```

### Erosion Prediction Model

```bash
python train_erosion_model.py
```

## Project Structure

```
.
├── dashboard.py              # Main Streamlit dashboard
├── inference_models.py      # Model loading and inference
├── train_yolo.py            # YOLOv5 training script
├── train_pillars_sculptures.py  # 4-class training script
├── train_erosion_model.py   # Erosion model training
├── dataset_yolo.yaml        # 2-class dataset config
├── dataset_yolo_pillars_sculptures.yaml  # 4-class dataset config
├── requirements_dashboard.txt  # Dashboard dependencies
└── runs/train/              # Training results and weights
```

## Model Performance

### Detection Model (30 epochs)
- **mAP@0.5**: 17.8%
- **Recall**: 60%
- **Classes**: Vegetation, Ruins

### Erosion Model
- **R² Score**: 0.9985
- **RMSE**: 0.0057

## Documentation

- [Dashboard README](DASHBOARD_README.md) - Dashboard usage guide
- [Pillars & Sculptures Setup](PILLARS_SCULPTURES_SETUP.md) - 4-class model setup
- [Quick Start Guide](QUICK_START_DASHBOARD.md) - Quick start instructions
- [YOLO Training Guide](README_YOLO.md) - YOLOv5 training details

## Usage Tips

### For Better Detection:
- Lower confidence threshold (0.05-0.1) for small objects
- Use larger inference size (832-1280) for better accuracy
- Train with more epochs (50-100) for improved performance

### For Pillars & Sculptures:
- Use the 4-class model configuration
- Lower confidence threshold to detect small objects
- See [PILLARS_SCULPTURES_SETUP.md](PILLARS_SCULPTURES_SETUP.md) for details

## Requirements

- Python 3.9+
- PyTorch
- Streamlit
- YOLOv5
- OpenCV
- NumPy, Pandas
- XGBoost, scikit-learn

See `requirements_dashboard.txt` for complete list.

## License

MIT License - see LICENSE file for details

## Contributors

- Uday Kumar

## Acknowledgments

- YOLOv5 by Ultralytics
- DeepLabV3 by PyTorch Vision
- Streamlit for the dashboard framework
