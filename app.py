import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from PIL import Image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Archaeological Site Analysis", layout="wide")

st.title("🏛 Archaeological Site Analysis System")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3
    )
    unet.load_state_dict(torch.load("models/unet_model.pth", map_location="cpu"))
    unet.eval()

    yolo = YOLO("models/archaeological_yolo_best.pt")
    return unet, yolo

unet_model, yolo_model = load_models()
st.success("Models loaded successfully")

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload an archaeological image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Input Image")
    st.image(image, use_container_width=True)

    # ===============================
    # U-NET SEGMENTATION
    # ===============================
    img_resized = cv2.resize(image_np, (512, 512))

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img_norm = (img_resized / 255.0 - mean) / std
    img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output = unet_model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().numpy()

    colored_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    colored_mask[pred_mask == 1] = [255, 0, 0]    # Ruins
    colored_mask[pred_mask == 2] = [0, 255, 0]    # Vegetation

    overlay = cv2.addWeighted(img_resized, 0.6, colored_mask, 0.8, 0)

    # ===============================
    # YOLO OBJECT DETECTION (FIXED)
    # ===============================
    yolo_results = yolo_model.predict(
        source=image_np,
        conf=0.05,          # 🔥 LOWER CONFIDENCE (IMPORTANT)
        iou=0.4,
        save=False
    )

    yolo_img = yolo_results[0].plot()
    yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)

    # ===============================
    # TERRAIN ANALYSIS (IMAGE-BASED)
    # ===============================
    green_channel = image_np[:, :, 1]
    vegetation_ratio = np.mean(green_channel) / 255.0

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    slope_score = np.mean(np.sqrt(gx**2 + gy**2))

    erosion_class = "Erosion-Prone" if slope_score > 80 else "Stable"

    # ===============================
    # DISPLAY RESULTS
    # ===============================
    st.subheader("Model Outputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("U-Net Segmentation")
        st.image(overlay, use_container_width=True)

    with col2:
        st.caption("YOLO Object Detection")
        st.image(yolo_img, use_container_width=True)

    with col3:
        st.caption("Terrain Analysis")
        st.metric("Vegetation Ratio", f"{vegetation_ratio:.4f}")
        st.metric("Slope Score", f"{slope_score:.2f}")
        st.metric("Erosion Risk", erosion_class)

    st.info(
        "ℹ YOLO uses a lower confidence threshold to visualize more potential archaeological structures. "
        "This may include false positives, which is expected for exploratory analysis."
    )
