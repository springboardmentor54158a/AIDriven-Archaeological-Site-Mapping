# AI-Driven Archaeological Site Mapping

## Project Statement
This project focuses on building an AI-based platform that analyzes satellite and drone imagery to assist archaeologists in identifying and preserving ancient sites. The system performs semantic segmentation of ruins and vegetation, detects archaeological structures, and predicts erosion-prone zones to support conservation planning and archaeological research.

---

## Objectives
- Segment ancient ruins and vegetation from satellite imagery
- Detect and classify archaeological structures
- Predict erosion-prone areas using terrain-related features
- Provide an integrated and visual analysis platform

---

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy, Pandas
- U-Net (Semantic Segmentation)
- YOLOv8 Nano (Object Detection)
- Random Forest (Erosion Prediction)
- Streamlit (Visualization)

---

## Project Modules

### 1. Semantic Segmentation (U-Net)
- Classes:
  - 0 → Background
  - 1 → Ruins
  - 2 → Vegetation
- Training Images: 138
- Validation Images: 4
- Dice Score: 0.764
- IoU Score: 0.661

---

### 2. Object Detection (YOLOv8)
- Model: YOLOv8 Nano
- Image Size: 640 × 640
- Epochs: 50
- Batch Size: 8
- Precision: 0.653
- Recall: 0.143
- mAP: 0.219

---

### 3. Terrain Erosion Prediction
- Features Used:
  - Vegetation Ratio
  - Slope Approximation
- Model: Random Forest
- RMSE: 0.0141
- R² Score: 0.9982
- Erosion Classification: Stable / Erosion-Prone

---

## System Integration
All trained models are integrated into a single Streamlit-based application that allows users to upload satellite images and view segmentation results, object detection outputs, and erosion risk predictions in real time.

---


## Author
Miza M  
B.Tech Information Technology – Final Year

---

## License
This project is intended for academic and research purposes.

