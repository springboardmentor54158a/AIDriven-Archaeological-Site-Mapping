# 🏺 AI-Driven Archaeological Site Mapping  
An AI platform for analyzing satellite and drone imagery to support archaeological research and conservation.

---

## 🌍 Project Overview

The **AI-Driven Archaeological Site Mapping** system uses deep learning and geospatial analytics to:

- 🧱 Segment ancient ruins and vegetation  
- 🎯 Detect and classify artifact structures  
- 🏜️ Predict terrain erosion zones  
- 📊 Visualize insights through an interactive dashboard  

This platform integrates semantic segmentation, object detection, and terrain modeling to support archaeologists in field analysis and conservation planning.

---

## 🎯 Project Outcomes

- Understand preprocessing of satellite/drone imagery  
- Build U-Net / DeepLabV3+ models for segmentation  
- Implement YOLOv5 / Faster R-CNN for artifact detection  
- Train XGBoost / Random Forest for erosion prediction  
- Deploy results through a Streamlit/Dash dashboard  

---

## 🗂️ Dataset Sources

- **Google Earth Pro**  
- **OpenAerialMap**  
- **Custom annotated datasets** (QGIS / Labelbox)  

---

## 🧩 Project Modules

1. Data Collection & Annotation  
2. Preprocessing & Augmentation  
3. Semantic Segmentation (Ruins & Vegetation)  
4. Object Detection & Artifact Classification  
5. Terrain Erosion Prediction  
6. Model Evaluation & Tuning  
7. Dashboard & Final Presentation  

---

## ⏳ Project Timeline

### 📌 Milestone 1: Dataset Collection & Preparation (Weeks 1–2)

**Week 1**  
- Download and review satellite/drone images  
- Define annotation schema (ruins, vegetation, artifacts)

**Week 2**  
- Annotate using Labelbox/QGIS  
- Normalize, resize, augment, and split the dataset  

---

### 📌 Milestone 2: Segmentation & Detection Models (Weeks 3–4)

**Week 3**  
- Implement U-Net / DeepLabV3+ for semantic segmentation  
- Validate with IoU & Dice Score  

**Week 4**  
- Train YOLOv5 / Faster R-CNN for artifact detection  
- Evaluate using mAP, precision, recall  

---

### 📌 Milestone 3: Terrain Erosion Prediction (Weeks 5–6)

**Week 5**  
- Collect terrain-related features (slope, NDVI, elevation, soil type)

**Week 6**  
- Train XGBoost / Random Forest  
- Evaluate using RMSE & R² Score  

---

### 📌 Milestone 4: Visualization & Reporting (Weeks 7–8)

**Week 7**  
- Build a Streamlit/Dash dashboard  
- Overlay segmentation, detection, and erosion layers on maps  

**Week 8**  
- Final documentation and write-up  
- Presentation and live project demo  

---

## 🔁 Workflow

1. Acquire + annotate imagery  
2. Preprocess and split datasets  
3. Train segmentation and detection models  
4. Predict erosion zones  
5. Visualize results on an interactive dashboard  

---

## 🛠️ Tech Stack

### Language  
- Python 🐍  

### Libraries  
- Pandas, NumPy  
- OpenCV, Rasterio  
- Scikit-learn  
- GeoPandas, Folium  
- Matplotlib, Seaborn  

### Deep Learning Frameworks  
- TensorFlow / Keras  
- PyTorch  

### Models  
- **Segmentation:** U-Net, DeepLabV3+  
- **Detection:** YOLOv5, Faster R-CNN  
- **Prediction:** XGBoost, Random Forest  

### Dashboard  
- Streamlit  
- Dash  

---

## 🏗️ Architecture

flowchart TD

A[Satellite / Drone Images] --> B[Data Preprocessing<br/>(Resizing, Normalization, Augmentation)]

B --> C1[Semantic Segmentation Model<br/>(U-Net / DeepLabV3+)]
B --> C2[Object Detection Model<br/>(YOLOv5 / Faster R-CNN)]
B --> C3[Terrain Feature Extraction<br/>(Slope, NDVI, Elevation)]

C3 --> D[Erosion Prediction Model<br/>(XGBoost / Random Forest)]

C1 --> E[Segmentation Output<br/>(Ruins / Vegetation Masks)]
C2 --> F[Artifact Detection Output<br/>(Bounding Boxes + Classes)]
D --> G[Erosion Risk Map]

E --> H[Interactive Dashboard<br/>(Streamlit / Dash)]
F --> H
G --> H

H --> I[Archaeological Insights & Map Visualization]


---

## 📊 Evaluation Metrics

### Segmentation  
- IoU (Intersection over Union)  
- Dice Score  

### Object Detection  
- mAP  
- Precision / Recall  

### Erosion Prediction  
- RMSE  
- R² Score  

---

## 📦 Final Deliverables

- Annotated dataset  
- Model training notebooks  
- Preprocessing scripts  
- Trained weights (optional)  
- Dashboard application  
- Final report & presentation  

---



