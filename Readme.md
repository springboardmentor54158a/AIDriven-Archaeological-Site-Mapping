# AI Driven Archaeological Site Mapping & Analysis System

## 📌 Project Overview
This project is an integrated Artificial Intelligence system designed to assist archaeologists in site discovery, artifact analysis, and environmental risk assessment. By combining Computer Vision (Object Detection and Semantic Segmentation) with Machine Learning (Regression Analysis), the system automates the processing of archaeological data. The final solution is deployed as an interactive web application using **Streamlit**, enabling real-time analysis of images and environmental parameters.

## 🚀 Key Features

### 1. 🏺 Artifact Detection (Object Detection)
*   **Model:** YOLOv5 (You Only Look Once)
*   **Functionality:** Automates the identification and classification of historical artifacts from images or live video feeds.
*   **Classes:** Coins, Jewelry, Pottery, Sculptures, Seals, Tablets, Weapons.
*   **Input:** Image upload, Camera capture, or Live Video feed.

### 2. 🗺️ Site Mapping (Semantic Segmentation)
*   **Model:** U-Net with ResNet34 encoder (Segmentation Models PyTorch)
*   **Functionality:** Analyzes aerial or satellite imagery to map archaeological sites by distinguishing man-made structures from natural terrain.
*   **Classes:**
    *   **Ruins:** Archaeological structures.
    *   **Vegetation:** Trees and plant cover.
    *   **Background:** Other terrain.
*   **Visualization:** Overlays predicted masks on original images for clear analysis and calculates class distribution percentages.

### 3. 🌍 Soil Erosion Risk Prediction
*   **Model:** Random Forest Regressor (Scikit-Learn)
*   **Functionality:** Predicts the risk of soil erosion at a specific site based on environmental parameters to prioritize conservation efforts.
*   **Input Parameters:**
    *   **Slope:** Terrain steepness (degrees).
    *   **NDVI:** Vegetation density index (-1 to 1).
    *   **Elevation (DEM):** Digital Elevation Model data (meters).
*   **Output:** Risk Score categorized as **Low**, **Medium**, or **High**.

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (for PyTorch on Windows)

### Steps
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install streamlit torch torchvision opencv-python matplotlib segmentation-models-pytorch joblib pandas pillow
    ```

3.  **Model Setup:**
    Ensure the trained model weights are placed in the correct directory (e.g., `models/`):
    *   `best.pt` (YOLOv5)
    *   `best_unet_week3_fixed.pth` (U-Net)
    *   `trained_regression_model.pkl` (Erosion Prediction)

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## 💻 Technologies Used
*   **Framework:** Streamlit
*   **Deep Learning:** PyTorch, Segmentation Models PyTorch (SMP)
*   **Computer Vision:** OpenCV, YOLOv5
*   **Machine Learning:** Scikit-Learn (Random Forest)
*   **Data Processing:** NumPy, Pandas
*   **Visualization:** Matplotlib, PIL

## ⚠️ Compatibility Notes
*   **NumPy 2.0:** The application includes patches to ensure compatibility with models trained using NumPy 2.0+ when running in environments with older NumPy versions.
*   **Windows Paths:** `pathlib` is patched to handle Linux-trained models (`PosixPath`) on Windows machines.
