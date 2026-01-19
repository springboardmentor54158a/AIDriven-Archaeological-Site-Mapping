import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Soil Erosion Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD CSS ----------------
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>🌍 AI-Driven Archaeological Site Mapping</h1>
    <p>UNet Segmentation • YOLO Detection • XGBoost Regression</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown('<div class="sidebar-title">🎛 Model Selection</div>', unsafe_allow_html=True)

if "model" not in st.session_state:
    st.session_state.model = "XGBoost"

if st.sidebar.button("🟢 XGBoost Regression"):
    st.session_state.model = "XGBoost"
if st.sidebar.button("🔵 UNet Segmentation"):
    st.session_state.model = "UNet"
if st.sidebar.button("🟡 YOLO Detection"):
    st.session_state.model = "YOLO"

model_choice = st.session_state.model

# ========================= XGBOOST =========================
if model_choice == "XGBoost":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🟢 XGBoost – Erosion Prediction")

    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="metric green">RMSE<br>0.0405</div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric blue">R² Score<br>0.9927</div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric orange">Dataset<br>Clean</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Actual vs Predicted")
    img = Path("outputs/xgboost_act_vs_pred_metrics/xgboost_actual_vs_predicted.png")
    if img.exists():
        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            st.image(str(img), width=420)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 NDVI + DEM Dataset")
    csv_path = Path("data/ndvi_dem_india_points.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        a,b,c = st.columns(3)
        a.markdown(f'<div class="metric purple">Rows<br>{df.shape[0]}</div>', unsafe_allow_html=True)
        b.markdown(f'<div class="metric blue">Columns<br>{df.shape[1]}</div>', unsafe_allow_html=True)
        c.markdown(f'<div class="metric red">Missing<br>{df.isnull().sum().sum()}</div>', unsafe_allow_html=True)

        if st.checkbox("Show CSV Preview"):
            st.dataframe(df.head(15), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= UNET =========================
elif model_choice == "UNet":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔵 UNet – Segmentation Results")

    images = [
        "outputs/unet/unet_1.png",
        "outputs/unet/unet_2.png",
        "outputs/unet/unet_3.png",
        "outputs/unet/unet_4.png",
        "outputs/unet/unet_5.png"
    ]

    cols = st.columns(3)
    for i,img in enumerate(images):
        cols[i%3].image(img, caption=f"Prediction {i+1}", use_container_width=True)

    # ✅ ONLY THIS PART IS CHANGED (ATTRACTIVE METRICS)
    metrics_path = Path("outputs/unet_metrics/evaluation_results.txt")
    if metrics_path.exists():
        with open(metrics_path) as f:
            lines = f.readlines()

        dice = ""
        iou = ""

        for line in lines:
            if "Dice" in line:
                dice = line.split(":")[-1].strip()
            if "IoU" in line:
                iou = line.split(":")[-1].strip()

        st.markdown("### 📊 Evaluation Metrics")
        m1, m2 = st.columns(2)
        m1.markdown(
            f'<div class="metric green">Average Dice Score<br>{dice}</div>',
            unsafe_allow_html=True
        )
        m2.markdown(
            f'<div class="metric blue">Average IoU Score<br>{iou}</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ========================= YOLO =========================
elif model_choice == "YOLO":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🟡 YOLO – Detection Performance")

    m1,m2,m3 = st.columns(3)
    m1.markdown('<div class="metric green">mAP<br>0.187</div>', unsafe_allow_html=True)
    m2.markdown('<div class="metric blue">Precision<br>0.601</div>', unsafe_allow_html=True)
    m3.markdown('<div class="metric orange">Recall<br>0.310</div>', unsafe_allow_html=True)

    st.divider()

    r1,r2,r3 = st.columns(3)
    r1.image("outputs/yolo/BoxP_curve.png", caption="Precision", use_container_width=True)
    r2.image("outputs/yolo/BoxR_curve.png", caption="Recall", use_container_width=True)
    r3.image("outputs/yolo/BoxF1_curve.png", caption="F1 Score", use_container_width=True)

    r4,r5 = st.columns(2)
    r4.image("outputs/yolo/BoxPR_curve.png", caption="PR Curve", use_container_width=True)
    r5.image("outputs/yolo/confusion_matrix_normalized.png", caption="Confusion Matrix", use_container_width=True)

    st.markdown("### 📈 YOLO Training Metrics")
    res = Path("outputs/yolo_metrics/results.png")
    if res.exists():
        st.image(str(res), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)