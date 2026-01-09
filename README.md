# AI-Driven Archaeological Site Mapping

## Project Overview
This project applies Artificial Intelligence and Machine Learning techniques
to analyze terrain data and images for predicting erosion risk, supporting
archaeological site conservation and planning.

The system integrates image-based deep learning models (YOLO / UNet) with
a structured machine learning workflow for training, inference, and result
export.

---

## Project Workflow

### 1. Dataset Preparation
- Image-based terrain datasets were used for erosion analysis.
- Images were resized and preprocessed before model training.
- No missing or corrupted data was observed.

---

### 2. Model Training
- Deep learning models (YOLO and UNet) were trained earlier using terrain images.
- Training experiments and results are stored in experiment and runs folders.
- A final training pipeline script was created to formalize the workflow.

**Script:**  
`src/train_model.py`

---

### 3. Model Inference (Prediction)
- Trained models were used to perform erosion risk prediction on unseen images.
- Predictions were generated for multiple terrain images.
- Results were exported in CSV format for analysis and visualization.

**Script:**  
`src/predict.py`

**Output:**  
`outputs/predictions.csv`

---

### 4. Risk Categorization
- Predicted erosion results were categorized into risk levels:
  - Low Risk
  - Medium Risk
  - High Risk
- This supports archaeological site risk assessment.

---

### 5. System Design
The project follows a modular structure:


src/ -> Training and prediction scripts
models/ -> Trained model files
outputs/ -> Prediction results
dataset/ -> Image datasets



---

## Technologies Used
- Python
- YOLO
- UNet
- OpenCV
- Scikit-learn
- Pandas

---

## Outcome
The system successfully demonstrates:
- End-to-end AI pipeline design
- Image-based erosion risk prediction
- Exportable results for GIS and further analysis

This project provides a practical AI solution for archaeological site monitoring
and conservation planning.
