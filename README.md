🏺 AI-Driven Archaeological Site Mapping

An AI platform for analyzing satellite and drone imagery to support archaeological research and conservation.

🌍 Project Overview

The AI-Driven Archaeological Site Mapping system uses deep learning and geospatial analytics to:

🧱 Segment ancient ruins and vegetation

🎯 Detect and classify artifact structures

🏜️ Predict terrain erosion zones

📊 Visualize insights through an interactive dashboard

This platform integrates semantic segmentation, object detection, and terrain modeling to support archaeologists in field analysis and conservation planning.

🎯 Project Outcomes

Understand preprocessing of satellite/drone imagery

Build U-Net / DeepLabV3+ models for segmentation

Implement YOLOv5 / Faster R-CNN for artifact detection

Train XGBoost / Random Forest for erosion prediction

Deploy results through a Streamlit/Dash dashboard

🗂️ Dataset Sources

Google Earth Pro

OpenAerialMap

Custom annotated images (QGIS / Labelbox)

🧩 Project Modules

Data Collection & Annotation

Preprocessing & Augmentation

Semantic Segmentation (Ruins & Vegetation)

Object Detection & Artifact Classification

Terrain Erosion Prediction

Model Evaluation & Tuning

Dashboard & Final Presentation

⏳ Project Timeline
📌 Milestone 1: Dataset Collection & Preparation (Weeks 1–2)

Week 1

Download satellite/drone images

Define annotation schema (ruins, vegetation, artifacts)

Week 2

Annotate using Labelbox/QGIS

Normalize, resize, and split dataset

📌 Milestone 2: Segmentation & Detection Models (Weeks 3–4)

Week 3

Implement U-Net / DeepLabV3+

Validate with IoU & Dice Score

Week 4

Train YOLOv5 / Faster R-CNN

Evaluate using mAP, precision, recall

📌 Milestone 3: Terrain Erosion Prediction (Weeks 5–6)

Week 5

Collect terrain features (slope, elevation, NDVI, etc.)

Week 6

Train XGBoost / Random Forest

Evaluate using RMSE & R² Score

📌 Milestone 4: Visualization & Reporting (Weeks 7–8)

Week 7

Build a Streamlit/Dash dashboard

Overlay segmentation, detection & erosion layers

Week 8

Final documentation

Presentation & demo

🔁 Workflow

Acquire + annotate imagery

Preprocess and split dataset

Train segmentation and detection models

Predict erosion zones

Visualize results on an interactive dashboard

🛠️ Tech Stack
Language

Python 🐍

Libraries

Pandas, NumPy

OpenCV, Rasterio

Scikit-learn

GeoPandas, Folium

Matplotlib, Seaborn

Deep Learning Frameworks

TensorFlow / PyTorch

Models

Segmentation: U-Net, DeepLabV3+

Detection: YOLOv5, Faster R-CNN

Prediction: XGBoost, Random Forest

Dashboard

Streamlit or Dash

🏗️ Architecture
[ Satellite/Drone Images ]
            ↓
[ Preprocessing & Augmentation ]
            ↓
 ┌──────────────────────────────────────────────┐
 │ Segmentation (U-Net/DeepLabV3+)              │
 │ Object Detection (YOLOv5/Faster R-CNN)       │
 │ Erosion Prediction (XGBoost/Random Forest)   │
 └──────────────────────────────────────────────┘
            ↓
[ Interactive Dashboard (Streamlit/Dash) ]
            ↓
[ Archaeological Insights & Map Visualizations ]

📊 Evaluation Metrics
Segmentation

IoU

Dice Score

Object Detection

mAP

Precision / Recall

Erosion Prediction

RMSE

R² Score

📦 Final Deliverables

Model training scripts

Annotated dataset

Trained model weights (optional)

Dashboard application

Final documentation & presentation
