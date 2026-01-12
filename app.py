import streamlit as st
import cv2
import numpy as np
import torch
import joblib
import segmentation_models_pytorch as smp
from ultralytics import YOLO

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Archaeological Site Analysis",
    layout="wide"
)

st.title("🏛️ Archaeological Site Analysis System")

# =====================================
# PATHS
# =====================================
UNET_MODEL_PATH = "models/unet_model.pth"
YOLO_MODEL_PATH = "models/archaeological_yolo_best.pt"
EROSION_MODEL_PATH = "models/TerrainErosion.pkl"  # optional if you use erosion

# =====================================
# LOAD MODELS (CACHED)
# =====================================
@st.cache_resource
def load_models():
    unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3
    )
    unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location="cpu"))
    unet.eval()

    yolo = YOLO(YOLO_MODEL_PATH)

    erosion = None
    try:
        erosion = joblib.load(EROSION_MODEL_PATH)
    except:
        pass

    return unet, yolo, erosion

unet_model, yolo_model, erosion_model = load_models()
st.success("✅ Models loaded successfully")

# =====================================
# IMAGE RESIZE (DISPLAY ONLY)
# =====================================
def resize_for_display(img, max_width=350):
    h, w, _ = img.shape
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# =====================================
# FILE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "Upload an archaeological image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    # ---------------------------------
    # READ IMAGE
    # ---------------------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # =====================================
    # INPUT IMAGE (SMALLER DISPLAY)
    # =====================================
    st.subheader("Input Image")
    st.image(resize_for_display(image_rgb, 350), width=350)

    # =====================================
    # U-NET SEGMENTATION
    # =====================================
    img_resized = cv2.resize(image_rgb, (512, 512))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img_norm = (img_resized / 255.0 - mean) / std
    img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output = unet_model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().numpy()

    seg_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    seg_mask[pred_mask == 1] = [255, 0, 0]   # Ruins
    seg_mask[pred_mask == 2] = [0, 255, 0]   # Vegetation

    overlay = cv2.addWeighted(img_resized, 0.6, seg_mask, 0.8, 0)

    # =====================================
    # YOLO OBJECT DETECTION
    # =====================================
    yolo_results = yolo_model.predict(
        source=image_rgb,
        conf=0.25,
        save=False
    )

    yolo_img = yolo_results[0].plot()
    yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)

    # =====================================
    # TERRAIN ANALYSIS
    # =====================================
    green_channel = image_rgb[:, :, 1]
    vegetation_ratio = np.mean(green_channel) / 255.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    slope_score = np.mean(np.sqrt(grad_x**2 + grad_y**2))

    erosion_class = "Stable"
    if erosion_model is not None:
        erosion_score = erosion_model.predict([[vegetation_ratio, slope_score]])[0]
        erosion_class = "Erosion-Prone" if erosion_score >= 0.5 else "Stable"

    # =====================================
    # OUTPUT SECTION
    # =====================================
    st.markdown("---")
    st.header("Model Outputs")

    col1, col2, col3 = st.columns([1, 1, 0.8])

    with col1:
        st.subheader("U-Net Segmentation")
        st.image(resize_for_display(overlay, 350), width=350)

    with col2:
        st.subheader("YOLO Object Detection")
        st.image(resize_for_display(yolo_img, 350), width=350)

    with col3:
        st.subheader("Terrain Analysis")
        st.metric("Vegetation Ratio", f"{vegetation_ratio:.4f}")
        st.metric("Slope Score", f"{slope_score:.2f}")
        st.metric("Erosion Risk", erosion_class)

    st.info(
        "ℹ️ YOLO uses a lower confidence threshold to visualize more potential "
        "archaeological structures. False positives are expected in exploratory analysis."
    )
