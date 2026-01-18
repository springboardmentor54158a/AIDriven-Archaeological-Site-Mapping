import streamlit as st
import os
import shutil
from PIL import Image
import numpy as np
import pathlib
import cv2
import matplotlib.pyplot as plt

# Try importing segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    st.error("segmentation_models_pytorch is not installed. Please run: pip install segmentation-models-pytorch")
    st.stop()

# Fix for loading models trained on Linux (PosixPath) on Windows
pathlib.PosixPath = pathlib.WindowsPath

# -----------------------------
# PyTorch import with Windows safety
# -----------------------------
try:
    import torch
except OSError as e:
    if "WinError 126" in str(e):
        st.error("PyTorch failed to load (WinError 126).")
        st.warning("Install Microsoft Visual C++ Redistributable.")
        st.markdown("[Download VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)")
        st.stop()
    else:
        raise e

import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="AI-Driven Site Mapping", layout="wide")
st.title("AI-Driven Site Mapping")
st.write(
    "Detect historical artifacts (coin, jewelry, pottery, sculpture, seal, tablet, weapon) "
    "using a YOLOv5 PyTorch model."
)

# -----------------------------
# PATHS (EDIT ONLY IF NEEDED)
# -----------------------------
YOLOV5_REPO_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\yolov5"
WEIGHTS_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\best.pt"
UNET_MODEL_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\best_unet_week3_fixed.pth"

# -----------------------------
# Validate paths
# -----------------------------
if not os.path.exists(os.path.join(YOLOV5_REPO_PATH, "hubconf.py")):
    st.error("YOLOv5 repository not found or hubconf.py missing.")
    st.stop()

if not os.path.exists(WEIGHTS_PATH):
    st.error("Model weights (best.pt) not found.")
    st.stop()

# -----------------------------
# Load YOLOv5 model (cached)
# -----------------------------
if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource
else:
    # Fallback for older Streamlit versions
    cache_resource = st.cache(allow_output_mutation=True)

@cache_resource
def load_model():
    model = torch.hub.load(
        YOLOV5_REPO_PATH,
        "custom",
        path=WEIGHTS_PATH,
        source="local"
    )
    model.conf = 0.25
    model.iou = 0.45
    return model

@cache_resource
def load_unet_model():
    if not os.path.exists(UNET_MODEL_PATH):
        raise FileNotFoundError(f"File not found: {UNET_MODEL_PATH}")
    try:
        # Load model using segmentation_models_pytorch as per the training configuration
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=3
        )
        state_dict = torch.load(UNET_MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load U-Net model: {e}")

def visualize_prediction(original_image, predicted_mask):
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap="viridis")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(original_image)
    plt.imshow(predicted_mask, alpha=0.5, cmap="jet")
    plt.axis("off")

    st.pyplot(fig)

def display_class_percentages(predicted_mask, num_classes=3):
    class_names = {
        0: "Background",
        1: "Ruins",
        2: "Vegetation"
    }
    total_pixels = predicted_mask.size
    st.subheader("Class Distribution")
    cols = st.columns(num_classes)
    for cls in range(num_classes):
        count = np.sum(predicted_mask == cls)
        percentage = (count / total_pixels) * 100
        label = class_names.get(cls, f"Class {cls}")
        cols[cls].metric(label=label, value=f"{percentage:.2f}%")

yolo_model = load_model()

try:
    unet_model = load_unet_model()
except Exception as e:
    st.warning(f"U-Net Model Warning: {e}")
    unet_model = None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Artifact Detection (YOLOv5)", "Site Mapping (U-Net)", "Erosion Prediction"])

with tab1:
    st.header("Artifact Detection")
    input_method = st.radio("Select Input Method", ["Upload Image", "Camera", "Live Video"], key="yolo_input")

    uploaded_file = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="yolo_uploader")
    elif input_method == "Camera":
        uploaded_file = st.camera_input("Take a picture", key="yolo_camera")

    if input_method == "Live Video":
        st.warning("Press 'Stop' to end the live feed.")
        run = st.checkbox('Run Live Detection', key="yolo_live")
        FRAME_WINDOW = st.image([])

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open video device.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture image from camera.")
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = yolo_model(frame_rgb, size=640)
                    results.render()
                    FRAME_WINDOW.image(results.ims[0])
                cap.release()

    elif uploaded_file is not None:
        st.success("Image received! Running detection...")
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        temp_dir = "temp_input"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(temp_image_path)

        try:
            results = yolo_model(temp_image_path, size=640)
            results.render()
            detected_img = results.ims[0]
            st.subheader("Detected Image")
            st.image(detected_img, use_column_width=True, channels="BGR")
            df = results.pandas().xyxy[0]
            if not df.empty:
                st.dataframe(df)
            else:
                st.info("No artifacts detected.")
        except Exception as e:
            st.error(f"Detection failed: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

with tab2:
    st.header("Site Mapping Segmentation")
    if unet_model is None:
        st.warning("Model failed to load. Check the error messages above.")
    else:
        unet_upload = st.file_uploader("Choose an image for segmentation", type=["jpg", "jpeg", "png"], key="unet_uploader")
        if unet_upload is not None:
            image = Image.open(unet_upload).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            if st.button("Segment Image"):
                with st.spinner("Running Segmentation..."):
                    try:
                        input_size = (256, 256)
                        img_resized = image.resize(input_size)
                        img_array = np.array(img_resized) / 255.0
                        
                        # Normalize using ImageNet stats (required for ResNet encoder)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_array = (img_array - mean) / std
                        
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
                        with torch.no_grad():
                            output = unet_model(img_tensor)
                        if output.shape[1] == 1:
                            pred_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
                        else:
                            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                        visualize_prediction(img_resized, pred_mask)
                        display_class_percentages(pred_mask, num_classes=3)
                    except Exception as e:
                        st.error(f"Segmentation failed: {e}")

with tab3:
    st.header("🌍 Soil Erosion Risk Prediction")
    
    st.markdown("---")
    
    # [ Input Parameters ]
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Slope (slider)
        slope = st.slider(
            "Slope (degrees)", 
            min_value=0.0, 
            max_value=90.0, 
            value=15.0,
            help="Steeper slopes increase water velocity and erosion risk."
        )
        
        # NDVI (slider)
        ndvi = st.slider(
            "NDVI (Vegetation Index)", 
            min_value=-1.0, 
            max_value=1.0, 
            value=0.4,
            help="Higher vegetation density (NDVI) stabilizes soil."
        )

    with col2:
        # Rainfall (number input)
        rainfall = st.number_input(
            "Annual Rainfall (mm)", 
            min_value=0.0, 
            value=1200.0,
            step=50.0
        )
        
        # Elevation (number input)
        elevation = st.number_input(
            "Elevation (meters)", 
            min_value=0.0, 
            value=500.0,
            step=10.0
        )

    st.markdown("---")

    # [ Predict Button ]
    if st.button("Predict Risk"):
        
        st.subheader("📊 Prediction Result")
        
        # --- HEURISTIC PREDICTION LOGIC ---
        # Logic: Risk increases with Slope & Rainfall, decreases with NDVI
        norm_slope = slope / 90.0
        norm_rainfall = min(rainfall, 3000) / 3000.0
        norm_ndvi = (ndvi + 1) / 2  # Shift -1..1 to 0..1
        
        risk_score = (0.5 * norm_slope) + (0.3 * norm_rainfall) - (0.4 * norm_ndvi)
        
        if risk_score < 0.0:
            st.success("🟢 LOW RISK\nThe combination of gentle slope and sufficient vegetation suggests the soil is stable.")
        elif risk_score < 0.2:
            st.warning("🟡 MEDIUM RISK\nThere is a moderate potential for soil loss. Monitoring is recommended.")
        else:
            st.error("🔴 HIGH RISK\nCritical conditions detected. Steep slope and high rainfall indicate severe erosion risk.")
