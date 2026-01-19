import os
import streamlit as st
from PIL import Image
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

from app_utils.yolo import load_yolo
from app_utils.unet import load_unet, prepare, postprocess
from app_utils.erosion import predict

st.set_page_config("AI Archaeology", layout="wide")
st.title("🏛️ AI Archaeological Mapping System")

BASE = os.getcwd()

@st.cache_resource
def load_models():
    yolo = load_yolo("yolov5", os.path.join("models","best.pt"))
    unet = load_unet(os.path.join("models","unet_model.pth"))
    erosion = joblib.load(os.path.join("models","erosion_class.pkl"))
    return yolo, unet, erosion

yolo, unet, erosion = load_models()

tab1, tab2, tab3 = st.tabs(["Artifacts", "Segmentation", "Erosion"])

# ------------------- Tab 1: YOLO Artifact Detection -------------------
with tab1:
    img_file = st.file_uploader("Upload image", ["jpg","png"])
    if img_file:
        im = Image.open(img_file).convert("RGB")
        results = yolo(np.array(im))
        results.render()  # draw boxes
        st.image(results.ims[0], caption="Detected Artifacts")

# ------------------- Tab 2: U-Net Segmentation -------------------
with tab2:
    img_file = st.file_uploader("Upload site image", ["jpg","png"], key="u")
    if img_file:
        im = Image.open(img_file).convert("RGB")
        input_tensor = prepare(im)  # preprocess
        with torch.no_grad():
            output = unet(input_tensor)

        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        mask_rgb = postprocess(mask)
        st.image(mask_rgb, caption="Vegetation Mask")

# ------------------- Tab 3: Erosion Prediction -------------------
with tab3:
    st.header("Soil Erosion Risk Prediction")

    # Input widgets
    elevation = st.number_input("Elevation (m)", value=500.0)
    ndvi = st.slider("NDVI Index", -1.0, 1.0, 0.3)
    slope = st.slider("Slope Angle (°)", 0.0, 90.0, 15.0)

    # Predict button
    if st.button("Predict Risk"):
        # Call the predict function with model and scaler
        risk = predict(erosion,slope=slope, dem=elevation, ndvi=ndvi)
        
        if risk == 0:
            risk = "Low"
        elif risk == 1:
            risk = "Moderate"
        else:
            risk = "High"
        st.metric("Predicted Soil Erosion Risk", f"{risk}")

