# import streamlit as st
# import os
# import shutil
# from PIL import Image
# import numpy as np
# import pathlib
# import cv2
# import matplotlib.pyplot as plt
# import joblib
# import sys
# import pandas as pd

# # Try importing segmentation_models_pytorch
# try:
#     import segmentation_models_pytorch as smp
# except ImportError:
#     st.error("segmentation_models_pytorch is not installed. Please run: pip install segmentation-models-pytorch")
#     st.stop()

# # -----------------------------
# # NumPy 2.0 Compatibility Patch
# # -----------------------------
# # Fix for "No module named 'numpy._core'" crashing the app
# if "numpy._core" not in sys.modules and hasattr(np, "core"):
#     sys.modules["numpy._core"] = np.core
# if "numpy._core.multiarray" not in sys.modules and hasattr(np, "core") and hasattr(np.core, "multiarray"):
#     sys.modules["numpy._core.multiarray"] = np.core.multiarray

# # Fix for loading models trained on Linux (PosixPath) on Windows
# pathlib.PosixPath = pathlib.WindowsPath

# # -----------------------------
# # PyTorch import with Windows safety
# # -----------------------------
# try:
#     import torch
# except OSError as e:
#     if "WinError 126" in str(e):
#         st.error("PyTorch failed to load (WinError 126).")
#         st.warning("Install Microsoft Visual C++ Redistributable.")
#         st.markdown("[Download VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)")
#         st.stop()
#     else:
#         raise e

# import torch.nn as nn
# import torch.nn.functional as F

# # -----------------------------
# # Streamlit config
# # -----------------------------
# st.set_page_config(page_title="AI Driven Archaeological Site Mapping ", layout="wide")
# st.title("AI Driven Archaeological Site Mapping ")


# # -----------------------------
# # PATHS (EDIT ONLY IF NEEDED)
# # -----------------------------
# YOLOV5_REPO_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\yolov5"
# WEIGHTS_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\best.pt"
# UNET_MODEL_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\best_unet_week3_fixed.pth"
# EROSION_MODEL_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\trained_regression_model.pkl"

# # -----------------------------
# # Validate paths
# # -----------------------------
# if not os.path.exists(os.path.join(YOLOV5_REPO_PATH, "hubconf.py")):
#     st.error("YOLOv5 repository not found or hubconf.py missing.")
#     st.stop()

# if not os.path.exists(WEIGHTS_PATH):
#     st.error("Model weights (best.pt) not found.")
#     st.stop()

# # -----------------------------
# # Load YOLOv5 model (cached)
# # -----------------------------
# if hasattr(st, "cache_resource"):
#     cache_resource = st.cache_resource
# else:
#     # Fallback for older Streamlit versions
#     cache_resource = st.cache(allow_output_mutation=True)

# @cache_resource
# def load_model():
#     model = torch.hub.load(
#         YOLOV5_REPO_PATH,
#         "custom",
#         path=WEIGHTS_PATH,
#         source="local"
#     )
#     model.conf = 0.25
#     model.iou = 0.45
#     return model

# @cache_resource
# def load_unet_model():
#     if not os.path.exists(UNET_MODEL_PATH):
#         raise FileNotFoundError(f"File not found: {UNET_MODEL_PATH}")
#     try:
#         # Load model using segmentation_models_pytorch as per the training configuration
#         model = smp.Unet(
#             encoder_name="resnet34",
#             encoder_weights=None,
#             in_channels=3,
#             classes=3
#         )
#         state_dict = torch.load(UNET_MODEL_PATH, map_location=torch.device('cpu'))
#         model.load_state_dict(state_dict)
#         model.eval()
#         return model
#     except Exception as e:
#         raise RuntimeError(f"Failed to load U-Net model: {e}")

# @cache_resource
# def load_erosion_model():
#     if not os.path.exists(EROSION_MODEL_PATH):
#         return None
#     try:
#         return joblib.load(EROSION_MODEL_PATH)
#     except Exception as e:
#         st.error(f"Error loading erosion model: {e}")
#         return None

# def visualize_prediction(original_image, predicted_mask):
#     fig = plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.title("Original Image")
#     plt.imshow(original_image)
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.title("Predicted Mask")
#     plt.imshow(predicted_mask, cmap="viridis")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.title("Overlay")
#     plt.imshow(original_image)
#     plt.imshow(predicted_mask, alpha=0.5, cmap="jet")
#     plt.axis("off")

#     st.pyplot(fig)

# def display_class_percentages(predicted_mask, num_classes=3):
#     class_names = {
#         0: "Background",
#         1: "Ruins",
#         2: "Vegetation"
#     }
#     total_pixels = predicted_mask.size
#     st.subheader("Class Distribution")
#     cols = st.columns(num_classes)
#     for cls in range(num_classes):
#         count = np.sum(predicted_mask == cls)
#         percentage = (count / total_pixels) * 100
#         label = class_names.get(cls, f"Class {cls}")
#         cols[cls].metric(label=label, value=f"{percentage:.2f}%")

# yolo_model = load_model()

# try:
#     unet_model = load_unet_model()
# except Exception as e:
#     st.warning(f"U-Net Model Warning: {e}")
#     unet_model = None

# # -----------------------------
# # Tabs
# # -----------------------------
# tab1, tab2, tab3 = st.tabs(["Artifact Detection (YOLOv5)", "Site Mapping (U-Net)", "Erosion Prediction"])

# with tab1:
#     st.header("Artifact Detection")
#     st.write(
#     "Detect historical artifacts (coin, jewelry, pottery, sculpture, seal, tablet, weapon) "
#     "using a YOLOv5 PyTorch model."
# )
#     input_method = st.radio("Select Input Method", ["Upload Image", "Camera", "Live Video"], key="yolo_input")

#     uploaded_file = None
#     if input_method == "Upload Image":
#         uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="yolo_uploader")
#     elif input_method == "Camera":
#         uploaded_file = st.camera_input("Take a picture", key="yolo_camera")

#     if input_method == "Live Video":
#         st.warning("Press 'Stop' to end the live feed.")
#         run = st.checkbox('Run Live Detection', key="yolo_live")
#         FRAME_WINDOW = st.image([])

#         if run:
#             cap = cv2.VideoCapture(0)
#             if not cap.isOpened():
#                 st.error("Could not open video device.")
#             else:
#                 while run:
#                     ret, frame = cap.read()
#                     if not ret:
#                         st.error("Failed to capture image from camera.")
#                         break
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     results = yolo_model(frame_rgb, size=640)
#                     results.render()
#                     FRAME_WINDOW.image(results.ims[0])
#                 cap.release()

#     elif uploaded_file is not None:
#         st.success("Image received! Running detection...")
#         image = Image.open(uploaded_file).convert("RGB")
#         st.subheader("Original Image")
#         st.image(image, use_column_width=True)

#         temp_dir = "temp_input"
#         os.makedirs(temp_dir, exist_ok=True)
#         temp_image_path = os.path.join(temp_dir, uploaded_file.name)
#         image.save(temp_image_path)

#         try:
#             results = yolo_model(temp_image_path, size=640)
#             results.render()
#             detected_img = results.ims[0]
#             st.subheader("Detected Image")
#             st.image(detected_img, use_column_width=True, channels="BGR")
#             df = results.pandas().xyxy[0]
#             if not df.empty:
#                 st.dataframe(df)
#             else:
#                 st.info("No artifacts detected.")
#         except Exception as e:
#             st.error(f"Detection failed: {e}")
#         finally:
#             shutil.rmtree(temp_dir, ignore_errors=True)

# with tab2:
#     st.header("Site Mapping Segmentation")
#     if unet_model is None:
#         st.warning("Model failed to load. Check the error messages above.")
#     else:
#         unet_upload = st.file_uploader("Choose an image for segmentation", type=["jpg", "jpeg", "png"], key="unet_uploader")
#         if unet_upload is not None:
#             image = Image.open(unet_upload).convert("RGB")
#             st.image(image, caption="Original Image", use_column_width=True)
#             if st.button("Segment Image"):
#                 with st.spinner("Running Segmentation..."):
#                     try:
#                         input_size = (256, 256)
#                         img_resized = image.resize(input_size)
#                         img_array = np.array(img_resized) / 255.0
                        
#                         # Normalize using ImageNet stats (required for ResNet encoder)
#                         mean = np.array([0.485, 0.456, 0.406])
#                         std = np.array([0.229, 0.224, 0.225])
#                         img_array = (img_array - mean) / std
                        
#                         img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
#                         with torch.no_grad():
#                             output = unet_model(img_tensor)
#                         if output.shape[1] == 1:
#                             pred_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
#                         else:
#                             pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
#                         visualize_prediction(img_resized, pred_mask)
#                         display_class_percentages(pred_mask, num_classes=3)
#                     except Exception as e:
#                         st.error(f"Segmentation failed: {e}")

# with tab3:
#     st.header("🌍 Soil Erosion Risk Prediction")
    
#     st.markdown("---")
    
#     # [ Input Parameters ]
#     st.subheader("Input Parameters")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Slope (slider)
#         slope = st.slider(
#             "Slope (degrees)", 
#             min_value=0.0, 
#             max_value=90.0, 
#             value=15.0,
#             help="Steeper slopes increase water velocity and erosion risk."
#         )
        
#         # NDVI (slider)
#         ndvi = st.slider(
#             "NDVI (Vegetation Index)", 
#             min_value=-1.0, 
#             max_value=1.0, 
#             value=0.4,
#             help="Higher vegetation density (NDVI) stabilizes soil."
#         )

#     with col2:
#         # Elevation (number input)
#         elevation = st.number_input(
#             "Elevation (meters)", 
#             min_value=0.0, 
#             value=500.0,
#             step=10.0
#         )

#     st.markdown("---")

#     # [ Predict Button ]
#     if st.button("Predict Risk"):
        
#         st.subheader("📊 Prediction Result")

#         with st.spinner("Loading model and calculating risk..."):
#             model = load_erosion_model()
            
#             if model is None:
#                 st.error("Could not load the model. Please check the file path or Python environment compatibility.")
#             else:
#                 try:
#                     # Prepare input vector as DataFrame matching training columns: ['DEM', 'NDVI', 'slope']
#                     input_features = pd.DataFrame({
#                         'DEM': [elevation],
#                         'NDVI': [ndvi],
#                         'slope': [slope]
#                     })
                    
#                     # Make prediction
#                     prediction = model.predict(input_features)
#                     risk_score = prediction[0]
                    
#                     st.metric("Predicted Risk Score", f"{risk_score:.4f}")
                    
#                     # Interpretation of the regression score
#                     if risk_score < 0.3:
#                         st.success("🟢 LOW RISK\nThe model predicts a low probability of soil erosion.")
#                     elif risk_score < 0.7:
#                         st.warning("🟡 MEDIUM RISK\nModerate erosion risk detected. Monitoring recommended.")
#                     else:
#                         st.error("🔴 HIGH RISK\nHigh erosion risk detected. Preventive measures advised.")
#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")


import streamlit as st
import os
import shutil
from PIL import Image
import numpy as np
import pathlib
import cv2
import matplotlib.pyplot as plt
import joblib
import sys
import pandas as pd
import time

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Archaeological Site Mapping AI",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom UI Styling (CSS)
# -----------------------------
st.markdown("""
<style>
    /* Main Background & Global Text Fix */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        color: #333333;
    }
    
    /* Ensure body text is visible (overrides Dark Mode defaults) */
    p, .stMarkdown, .stText, .stCaption, li {
        color: #333333 !important;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2, h3 {
        color: #5D4037; /* Darker Earth Tone (Coffee) for better readability */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Widget Labels */
    .stRadio label, .stCheckbox label, .stFileUploader label, .stSlider label, .stNumberInput label {
        color: #333333 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #8B4513;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #A0522D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #5D4037;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #555555; /* Default tab text color */
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 2px solid #8B4513;
        color: #8B4513 !important; /* Active tab text color */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Imports & Logic Protection
# -----------------------------

# Try importing segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    st.error("🚨 segmentation_models_pytorch is not installed. Please run: `pip install segmentation-models-pytorch`")
    st.stop()

# NumPy 2.0 Compatibility Patch
if "numpy._core" not in sys.modules and hasattr(np, "core"):
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules and hasattr(np, "core") and hasattr(np.core, "multiarray"):
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

# Fix for loading models trained on Linux (PosixPath) on Windows
pathlib.PosixPath = pathlib.WindowsPath

# PyTorch import with Windows safety
try:
    import torch
except OSError as e:
    if "WinError 126" in str(e):
        st.error("🚨 PyTorch failed to load (WinError 126).")
        st.warning("Install Microsoft Visual C++ Redistributable.")
        st.markdown("[Download VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)")
        st.stop()
    else:
        raise e

import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Header Section
# -----------------------------
col_h1, col_h2, col_h3 = st.columns([1, 6, 1])
with col_h2:
    st.title("🏺 AI Driven Archaeological Site Mapping")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Advanced artifact detection, site segmentation, and soil erosion risk analysis.</p>", 
        unsafe_allow_html=True
    )

# -----------------------------
# PATHS (UNCHANGED LOGIC)
# -----------------------------
YOLOV5_REPO_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\yolov5"
WEIGHTS_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\best.pt"
UNET_MODEL_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\best_unet_week3_fixed.pth"
EROSION_MODEL_PATH = r"C:\Users\ALSHIFANA\Downloads\final models\models\trained_regression_model.pkl"

# -----------------------------
# Validate paths
# -----------------------------
path_errors = []
if not os.path.exists(os.path.join(YOLOV5_REPO_PATH, "hubconf.py")):
    path_errors.append(f"YOLOv5 repository not found at: `{YOLOV5_REPO_PATH}`")

if not os.path.exists(WEIGHTS_PATH):
    path_errors.append(f"Model weights not found at: `{WEIGHTS_PATH}`")

if path_errors:
    for err in path_errors:
        st.error(err)
    st.stop()

# -----------------------------
# Model Loading Functions
# -----------------------------
if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource
else:
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

@cache_resource
def load_erosion_model():
    if not os.path.exists(EROSION_MODEL_PATH):
        return None
    try:
        return joblib.load(EROSION_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading erosion model: {e}")
        return None

# -----------------------------
# Visualization Utilities
# -----------------------------
def visualize_prediction(original_image, predicted_mask):
    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.imshow(original_image)
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Predicted Mask", fontsize=12, fontweight='bold')
    ax2.imshow(predicted_mask, cmap="viridis")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Overlay Analysis", fontsize=12, fontweight='bold')
    ax3.imshow(original_image)
    ax3.imshow(predicted_mask, alpha=0.5, cmap="jet")
    ax3.axis("off")

    st.pyplot(fig)

def display_class_percentages(predicted_mask, num_classes=3):
    class_names = {
        0: "Background",
        1: "Ruins",
        2: "Vegetation"
    }
    total_pixels = predicted_mask.size
    
    st.markdown("#### Class Distribution Analysis")
    cols = st.columns(num_classes)
    
    for cls in range(num_classes):
        count = np.sum(predicted_mask == cls)
        percentage = (count / total_pixels) * 100
        label = class_names.get(cls, f"Class {cls}")
        
        with cols[cls]:
            st.container()
            st.metric(label=label, value=f"{percentage:.2f}%")

# -----------------------------
# Load Models
# -----------------------------
with st.spinner("Initializing AI Models..."):
    yolo_model = load_model()
    try:
        unet_model = load_unet_model()
    except Exception as e:
        st.warning(f"⚠️ U-Net Model Warning: {e}")
        unet_model = None

# -----------------------------
# Main Application Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Artifact Detection", "🗺️ Site Mapping", "⛰️ Erosion Prediction"])

# --- TAB 1: Artifact Detection ---
with tab1:
    st.header("Artifact Detection")
    st.info("Detect historical artifacts: Coin, Jewelry, Pottery, Sculpture, Seal, Tablet, Weapon")

    # Input Method Selection in a cleaner layout
    col_input, col_display = st.columns([1, 2])

    with col_input:
        st.subheader("Configuration")
        input_method = st.radio(
            "Select Input Source", 
            ["Upload Image", "Camera", "Live Video"], 
            key="yolo_input"
        )

    with col_display:
        uploaded_file = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("📂 Choose an image", type=["jpg", "jpeg", "png"], key="yolo_uploader")
        
        elif input_method == "Camera":
            uploaded_file = st.camera_input("📷 Take a picture", key="yolo_camera")

        elif input_method == "Live Video":
            st.warning("Press 'Stop' to end the live feed.")
            run = st.checkbox('🔴 Run Live Detection', key="yolo_live")
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

    # Process Uploaded/Camera Image
    if uploaded_file is not None and input_method != "Live Video":
        st.markdown("---") # Replaced st.divider()
        st.subheader("Detection Results")
        
        image = Image.open(uploaded_file).convert("RGB")
        
        temp_dir = "temp_input"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(temp_image_path)

        try:
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.image(image, caption="Original Image", use_column_width=True)

            with st.spinner("Detecting artifacts..."):
                results = yolo_model(temp_image_path, size=640)
                results.render()
                detected_img = results.ims[0]
                
            with col_res2:
                st.image(detected_img, caption="AI Detection", use_column_width=True, channels="BGR")

            df = results.pandas().xyxy[0]
            if not df.empty:
                st.success(f"Detected {len(df)} artifacts!")
                with st.expander("View Detailed Data"):
                    # Removed use_container_width=True for compatibility
                    st.dataframe(df)
            else:
                st.info("No artifacts detected in this image.")
                
        except Exception as e:
            st.error(f"Detection failed: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

# --- TAB 2: Site Mapping ---
with tab2:
    st.header("Site Mapping Segmentation")
    st.markdown("Use deep learning (U-Net) to segment aerial imagery into **Ruins**, **Vegetation**, and **Background**.")

    if unet_model is None:
        st.error("❌ Model failed to load. Check the error messages at the top.")
    else:
        unet_upload = st.file_uploader("📂 Choose an image for segmentation", type=["jpg", "jpeg", "png"], key="unet_uploader")
        
        if unet_upload is not None:
            image = Image.open(unet_upload).convert("RGB")
            
            col_preview, col_action = st.columns([1, 2])
            with col_preview:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col_action:
                st.write("Ready to process.")
                if st.button("✨ Run Segmentation Analysis"):
                    with st.spinner("Processing image (Resizing -> Normalizing -> U-Net Inference)..."):
                        try:
                            input_size = (256, 256)
                            img_resized = image.resize(input_size)
                            img_array = np.array(img_resized) / 255.0
                            
                            # Normalize using ImageNet stats
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
                                
                            st.success("Segmentation Complete!")
                            
                        except Exception as e:
                            st.error(f"Segmentation failed: {e}")
                            pred_mask = None

            if 'pred_mask' in locals() and pred_mask is not None:
                st.markdown("---") # Replaced st.divider()
                visualize_prediction(img_resized, pred_mask)
                display_class_percentages(pred_mask, num_classes=3)

# --- TAB 3: Erosion Prediction ---
with tab3:
    st.header("🌍 Soil Erosion Risk Prediction")
    st.markdown("Predict the risk of soil erosion based on topographical and environmental factors.")
    
    st.markdown("---") # Replaced st.divider()
    
    # Create a nice card-like container for inputs
    with st.container():
        st.subheader("📝 Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Topography**")
            slope = st.slider(
                "Slope (degrees)", 
                min_value=0.0, max_value=90.0, value=15.0,
                help="Steeper slopes increase water velocity and erosion risk."
            )
            elevation = st.number_input(
                "Elevation (meters)", 
                min_value=0.0, value=500.0, step=10.0
            )

        with col2:
            st.markdown("**Vegetation**")
            ndvi = st.slider(
                "NDVI (Vegetation Index)", 
                min_value=-1.0, max_value=1.0, value=0.4,
                help="Normalized Difference Vegetation Index. Higher values indicate denser vegetation (-1 to 1)."
            )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Removed use_container_width=True for compatibility
    if st.button("🚀 Predict Risk Score"):
        st.markdown("---") # Replaced st.divider()
        st.subheader("📊 Analysis Result")

        with st.spinner("Calculating risk model..."):
            model = load_erosion_model()
            
            if model is None:
                st.error("❌ Could not load the erosion model. Please check the file path.")
            else:
                try:
                    # Prepare input vector
                    input_features = pd.DataFrame({
                        'DEM': [elevation],
                        'NDVI': [ndvi],
                        'slope': [slope]
                    })
                    
                    # Make prediction
                    prediction = model.predict(input_features)
                    risk_score = prediction[0]
                    
                    # Layout for results
                    col_res_metric, col_res_interp = st.columns([1, 2])
                    
                    with col_res_metric:
                        st.metric("Predicted Risk Score", f"{risk_score:.4f}")
                    
                    with col_res_interp:
                        # Interpretation
                        if risk_score < 0.3:
                            st.success("🟢 **LOW RISK**\nThe model predicts a low probability of soil erosion.")
                        elif risk_score < 0.7:
                            st.warning("🟡 **MEDIUM RISK**\nModerate erosion risk detected. Monitoring recommended.")
                        else:
                            st.error("🔴 **HIGH RISK**\nHigh erosion risk detected. Preventive measures advised.")
                            
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("AI Driven Archaeological Site Mapping © 2024 | Powered by YOLOv5, U-Net & Streamlit")