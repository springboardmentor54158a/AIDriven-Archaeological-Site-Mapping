"""
Archaeological Analysis Dashboard
Interactive Streamlit dashboard integrating YOLO detection, U-Net segmentation, and erosion prediction
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time

# Configure page
st.set_page_config(
    page_title="Archaeological Analysis Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models safely
@st.cache_resource
def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(str(path))

@st.cache_resource
def load_unet(path):
    import torch
    import segmentation_models_pytorch as smp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_erosion_model(path):
    import joblib
    return joblib.load(path)

# Header
st.markdown('<h1 class="main-header">🏛️ Archaeological Analysis Dashboard</h1>', unsafe_allow_html=True)
base_dir = Path(__file__).parent

# Upload Section
uploaded_file = st.file_uploader("Upload an image to start analysis...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 1. Image Loading
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    
    # Placeholders for results
    results = {
        "yolo": {"count": 0, "img": None, "df": None, "error": None},
        "unet": {"ruins_pct": 0.0, "veg_pct": 0.0, "back_pct": 0.0, "mask": None, "error": None},
        "erosion": {"risk_label": "Unknown", "risk_color": "#999", "prob": 0.0, "veg_idx": 0.0, "slope": 0.0, "error": None}
    }

    # PROCEED WITH ANALYSIS (Showing Progress)
    with st.status("Processing Image...", expanded=True) as status:
        
        # --- YOLO Analysis ---
        st.write("🔍 Running YOLO Detection...")
        yolo_path = base_dir / "archaeological_yolo_best.pt"
        if yolo_path.exists():
            try:
                model = load_yolo(yolo_path)
                y_results = model.predict(img_array, conf=0.25) # Default conf
                
                annotated = y_results[0].plot()
                results["yolo"]["img"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                detections = []
                for box in y_results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = y_results[0].names[cls_id]
                    detections.append({'Class': cls_name, 'Confidence': f"{conf:.2%}"})
                
                results["yolo"]["df"] = pd.DataFrame(detections)
                results["yolo"]["count"] = len(detections)
            except Exception as e:
                results["yolo"]["error"] = str(e)
        else:
            results["yolo"]["error"] = "Model not found"

        # --- U-Net Analysis ---
        st.write("🗺️ Running Segmentation...")
        unet_path = base_dir / "unet_archaeology.pth"
        if unet_path.exists():
            try:
                model, device = load_unet(unet_path)
                
                img_resized = cv2.resize(img_array, (512, 512))
                img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(img_tensor)
                    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
                
                results["unet"]["mask"] = pred_mask
                total = pred_mask.size
                results["unet"]["ruins_pct"] = np.sum(pred_mask == 1) / total
                results["unet"]["veg_pct"] = np.sum(pred_mask == 2) / total
                results["unet"]["back_pct"] = np.sum(pred_mask == 0) / total
            except Exception as e:
                results["unet"]["error"] = str(e)
        else:
            results["unet"]["error"] = "Model not found"

        # --- Erosion Analysis ---
        st.write("⛰️ Predicting Erosion Risk...")
        erosion_path = base_dir / "erosion_model.pkl"
        if erosion_path.exists():
            try:
                model = load_erosion_model(erosion_path)
                
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_resized = cv2.resize(img_bgr, (512, 512))
                
                veg = np.mean(img_resized[:, :, 1]) / 255.0
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                slope = np.mean(np.sqrt(gx**2 + gy**2)) / 255.0
                
                feat = np.array([[veg, slope]])
                pred = model.predict(feat)[0]
                prob = model.predict_proba(feat)[0]
                
                results["erosion"]["veg_idx"] = veg
                results["erosion"]["slope"] = slope
                
                if pred == 1:
                    results["erosion"]["risk_label"] = "High Risk"
                    results["erosion"]["risk_color"] = "#FF6B6B"
                    results["erosion"]["prob"] = prob[1] * 100
                else:
                    results["erosion"]["risk_label"] = "Stable"
                    results["erosion"]["risk_color"] = "#51CF66"
                    results["erosion"]["prob"] = prob[1] * 100
            except Exception as e:
                results["erosion"]["error"] = str(e)
        else:
            results["erosion"]["error"] = "Model not found"
            
        status.update(label="Analysis Complete", state="complete", expanded=False)

    # --- RENDER RESULTS ---
    st.markdown("---")
    
    # 1. Summary Metrics Display (Top Row)
    m1, m2, m3 = st.columns(3)
    
    with m1:
        count = results["yolo"]["count"]
        st.markdown(f"""<div class="metric-card"><h4>YOLO Detections</h4><h2>{count}</h2></div>""", unsafe_allow_html=True)
        
    with m2:
        rpct = results["unet"]["ruins_pct"]
        st.markdown(f"""<div class="metric-card"><h4>Ruins Coverage</h4><h2>{rpct:.1%}</h2></div>""", unsafe_allow_html=True)
        
    with m3:
        rlabel = results["erosion"]["risk_label"]
        rcolor = results["erosion"]["risk_color"]
        st.markdown(f"""<div class="metric-card" style="border: 2px solid {rcolor}"><h4>Erosion Status</h4><h2 style="color:{rcolor}">{rlabel}</h2></div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. Detailed Views
    tab1, tab2, tab3 = st.tabs(["🔍 Detection Details", "🗺️ Segmentation Details", "⛰️ Erosion Details"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, width='stretch', caption="Original Image")
        with col2:
            if results["yolo"]["error"]:
                st.error(results["yolo"]["error"])
            elif results["yolo"]["img"] is not None:
                st.image(results["yolo"]["img"], width='stretch', caption="YOLO Detections")
                if results["yolo"]["df"] is not None and not results["yolo"]["df"].empty:
                    st.dataframe(results["yolo"]["df"], width='stretch')
            else:
                st.info("No detections or model not run.")

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, width='stretch', caption="Original Image")
        with col2:
            if results["unet"]["error"]:
                st.error(results["unet"]["error"])
            elif results["unet"]["mask"] is not None:
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(["#d3d3d3", "#FF6B6B", "#51CF66"])
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(results["unet"]["mask"], cmap=cmap, vmin=0, vmax=2)
                ax.axis('off')
                st.pyplot(fig)
                
                # Composition Bar
                fig_bar = go.Figure(data=[
                    go.Bar(name='Ruins', x=['Coverage'], y=[results["unet"]["ruins_pct"]], marker_color='#FF6B6B'),
                    go.Bar(name='Vegetation', x=['Coverage'], y=[results["unet"]["veg_pct"]], marker_color='#51CF66'),
                    go.Bar(name='Background', x=['Coverage'], y=[results["unet"]["back_pct"]], marker_color='#d3d3d3')
                ])
                fig_bar.update_layout(barmode='stack', height=200, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_bar, width='stretch')

    with tab3:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, width='stretch', caption="Original Image")
        with col2:
            if results["erosion"]["error"]:
                st.error(results["erosion"]["error"])
            else:
                st.metric("Vegetation Index", f"{results['erosion']['veg_idx']:.3f}")
                st.metric("Slope Score", f"{results['erosion']['slope']:.3f}")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results["erosion"]["prob"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Erosion Probability (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': results["erosion"]["risk_color"]},
                        'steps': [
                            {'range': [0, 50], 'color': "#e6ffe6"},
                            {'range': [50, 100], 'color': "#ffe6e6"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_gauge, width='stretch')

# Footer
st.markdown("---")
st.markdown("ArchDashboard v2.1 | Powered by YOLO & U-Net")
