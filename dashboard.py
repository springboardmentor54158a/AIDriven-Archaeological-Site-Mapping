"""
Interactive Dashboard for Ruins & Vegetation Analysis
Integrates: Segmentation, Detection, and Erosion Prediction
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Ruins & Vegetation Analysis Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏛️ Ruins & Vegetation Analysis Dashboard</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### Model Settings")

# Model paths
seg_model_path = st.sidebar.text_input("Segmentation Model Path (optional)", 
                                       value="", 
                                       help="Leave empty to use pretrained model")
det_model_path = st.sidebar.text_input("Detection Model Path", 
                                       value="runs/train/artifacts_improved/weights/best.pt",
                                       help="Path to YOLOv5 weights. Using improved model trained for 30 epochs")
ero_model_path = st.sidebar.text_input("Erosion Model Path", 
                                       value="erosion_xgboost_regression.pkl",
                                       help="Path to erosion prediction model")

# Detection settings
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.1, 0.01, 
                                   help="Lower values detect more objects but may include false positives. Recommended: 0.05-0.15 for small objects like pillars")
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
inference_size = st.sidebar.selectbox("Inference Image Size", [416, 640, 832, 1280], index=1,
                                     help="Larger sizes detect smaller objects better but are slower")

# Device selection
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
if device == "cuda" and not torch.cuda.is_available():
    st.sidebar.warning("CUDA not available, using CPU")
    device = "cpu"

# Load models (cached)
@st.cache_resource
def load_all_models(seg_path, det_path, ero_path, device):
    """Load all models"""
    from inference_models import load_segmentation_model, load_detection_model, load_erosion_model
    
    models = {}
    
    # Segmentation
    try:
        models['segmentation'] = load_segmentation_model(seg_path if seg_path else None, device)
        st.sidebar.success("✓ Segmentation model loaded")
    except Exception as e:
        st.sidebar.error(f"✗ Segmentation model failed: {e}")
        models['segmentation'] = None
    
    # Detection
    try:
        models['detection'] = load_detection_model(det_path, device)
        st.sidebar.success("✓ Detection model loaded")
    except Exception as e:
        st.sidebar.error(f"✗ Detection model failed: {e}")
        models['detection'] = None
    
    # Erosion
    try:
        models['erosion'] = load_erosion_model(ero_path)
        st.sidebar.success("✓ Erosion model loaded")
    except Exception as e:
        st.sidebar.error(f"✗ Erosion model failed: {e}")
        models['erosion'] = None
    
    return models

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📊 Batch Analysis", "ℹ️ About"])

with tab1:
    st.header("Image Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    # Or select from existing images
    col1, col2 = st.columns(2)
    with col1:
        use_existing = st.checkbox("Use existing image from dataset")
    with col2:
        if use_existing:
            image_files = list(Path("images").glob("*.jpg")) + list(Path("images").glob("*.jpeg"))
            if image_files:
                selected_image = st.selectbox("Select image", 
                                             [str(f) for f in image_files])
            else:
                selected_image = None
                st.warning("No images found in 'images' directory")
        else:
            selected_image = None
    
    # Process image
    if uploaded_file is not None or selected_image:
        # Load image
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            image_name = uploaded_file.name
            temp_path = f"temp_{image_name}"
            image.save(temp_path)
        else:
            image = Image.open(selected_image).convert('RGB')
            image_name = Path(selected_image).name
            temp_path = selected_image
        
        # Display original
        st.subheader("Original Image")
        st.image(image)
        
        # Load models
        with st.spinner("Loading models..."):
            models = load_all_models(seg_model_path, det_model_path, ero_model_path, device)
        
        # Process
        if any(models.values()):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Segmentation", "✓" if models['segmentation'] else "✗")
            with col2:
                st.metric("Detection", "✓" if models['detection'] else "✗")
            with col3:
                st.metric("Erosion", "✓" if models['erosion'] else "✗")
            
            # Warning about model limitations
            if models['detection']:
                st.info("⚠️ **Model Performance Note**: \n"
                       "**For pillars and sculptures detection:**\n"
                       "- Use the 4-class model (pillars/sculptures) - see PILLARS_SCULPTURES_SETUP.md\n"
                       "- Lower confidence threshold (0.05-0.1)\n"
                       "- Larger inference size (832-1280)\n"
                       "- Current model may have limited performance on small objects")
            
            # Run inference
            with st.spinner("Running analysis..."):
                from inference_models import (predict_segmentation, predict_detection, 
                                             predict_erosion, create_unified_visualization)
                
                results = {}
                
                # Segmentation
                if models['segmentation']:
                    seg_mask = predict_segmentation(models['segmentation'], image, device)
                    results['segmentation'] = seg_mask
                else:
                    results['segmentation'] = None
                
                # Detection
                if models['detection']:
                    detections, annotated_img = predict_detection(models['detection'], 
                                                                 image, conf_threshold, 
                                                                 inference_size=inference_size,
                                                                 iou_threshold=iou_threshold)
                    results['detections'] = detections
                    results['annotated_img'] = annotated_img
                else:
                    results['detections'] = None
                    results['annotated_img'] = None
                
                # Erosion
                if models['erosion']:
                    erosion_prob, feature_info = predict_erosion(models['erosion'], temp_path)
                    results['erosion'] = erosion_prob
                    results['erosion_features'] = feature_info
                else:
                    results['erosion'] = None
                    results['erosion_features'] = None
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create unified visualization
            # Ensure segmentation mask matches image size
            seg_mask = results['segmentation']
            if seg_mask is not None:
                img_array = np.array(image)
                if seg_mask.shape[:2] != img_array.shape[:2]:
                    # Resize mask to match image if needed
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray(seg_mask.astype(np.uint8), mode='L')
                    mask_resized = mask_pil.resize((img_array.shape[1], img_array.shape[0]), PILImage.NEAREST)
                    seg_mask = np.array(mask_resized)
            else:
                img_array = np.array(image)
                seg_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            
            fig = create_unified_visualization(
                img_array,
                seg_mask,
                results['detections'],
                results['erosion']
            )
            st.pyplot(fig)
            
            # Detailed results
            st.subheader("Detailed Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Segmentation stats
            if results['segmentation'] is not None:
                seg_mask = results['segmentation']
                # Ensure mask matches image size for accurate calculation
                img_array = np.array(image)
                if seg_mask.shape[:2] != img_array.shape[:2]:
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray(seg_mask.astype(np.uint8), mode='L')
                    mask_resized = mask_pil.resize((img_array.shape[1], img_array.shape[0]), PILImage.NEAREST)
                    seg_mask = np.array(mask_resized)
                
                total_pixels = seg_mask.size
                veg_pixels = np.sum(seg_mask == 1)
                ruins_pixels = np.sum(seg_mask == 2)
                bg_pixels = np.sum(seg_mask == 0)
                
                with col1:
                    st.metric("Vegetation Coverage", f"{(veg_pixels/total_pixels)*100:.1f}%")
                with col2:
                    st.metric("Ruins Coverage", f"{(ruins_pixels/total_pixels)*100:.1f}%")
            else:
                with col1:
                    st.metric("Vegetation Coverage", "N/A")
                with col2:
                    st.metric("Ruins Coverage", "N/A")
            
            # Detection stats
            if results['detections'] is not None and len(results['detections']) > 0:
                detections_df = results['detections']
                with col3:
                    st.metric("Objects Detected", len(detections_df))
                with col4:
                    if 'class' in detections_df.columns:
                        veg_det = len(detections_df[detections_df['class'] == 0])
                        ruins_det = len(detections_df[detections_df['class'] == 1])
                        pillars_det = len(detections_df[detections_df['class'] == 2])
                        sculptures_det = len(detections_df[detections_df['class'] == 3])
                        det_text = f"{veg_det}V/{ruins_det}R"
                        if pillars_det > 0 or sculptures_det > 0:
                            det_text += f"/{pillars_det}P/{sculptures_det}S"
                        st.metric("Detections", det_text)
                    else:
                        st.metric("Detections", "N/A")
            else:
                with col3:
                    st.metric("Objects Detected", "0")
                with col4:
                    st.metric("Detections", "0")
            
            # Erosion stats
            if results['erosion'] is not None:
                st.subheader("Erosion Prediction")
                erosion_prob = results['erosion']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Erosion Probability", f"{erosion_prob:.3f}")
                    if erosion_prob < 0.3:
                        st.success("✅ Stable - Low erosion risk")
                    elif erosion_prob < 0.6:
                        st.warning("⚠️ Moderate Risk - Monitor area")
                    else:
                        st.error("🚨 High Risk - Erosion-prone area")
                
                with col2:
                    if results['erosion_features']:
                        st.write("**Feature Values:**")
                        st.write(f"- Vegetation Index: {results['erosion_features']['vegetation_index']:.3f}")
                        st.write(f"- Slope Proxy: {results['erosion_features']['slope_proxy']:.3f}")
                        st.write(f"- Texture Variance: {results['erosion_features']['texture_variance']:.3f}")
            
            # Cleanup temp file
            if uploaded_file and os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            st.error("No models loaded. Please check model paths in sidebar.")

with tab2:
    st.header("Batch Analysis")
    
    st.info("Upload multiple images or select a directory to analyze all images at once.")
    
    batch_files = st.file_uploader("Upload multiple images", type=['jpg', 'jpeg', 'png'], 
                                   accept_multiple_files=True)
    
    if batch_files:
        st.write(f"Processing {len(batch_files)} images...")
        
        # Load models once
        with st.spinner("Loading models..."):
            models = load_all_models(seg_model_path, det_model_path, ero_model_path, device)
        
        progress_bar = st.progress(0)
        results_summary = []
        
        for idx, uploaded_file in enumerate(batch_files):
            image = Image.open(uploaded_file).convert('RGB')
            temp_path = f"temp_batch_{uploaded_file.name}"
            image.save(temp_path)
            
            # Run analysis
            from inference_models import (predict_segmentation, predict_detection, 
                                         predict_erosion)
            
            result = {'image': uploaded_file.name}
            
            if models['segmentation']:
                seg_mask = predict_segmentation(models['segmentation'], image, device)
                total_pixels = seg_mask.size
                result['veg_coverage'] = (np.sum(seg_mask == 1) / total_pixels) * 100
                result['ruins_coverage'] = (np.sum(seg_mask == 2) / total_pixels) * 100
            
            if models['detection']:
                detections, _ = predict_detection(models['detection'], image, conf_threshold, 
                                                 inference_size=inference_size, iou_threshold=iou_threshold)
                result['objects_detected'] = len(detections) if detections is not None and len(detections) > 0 else 0
            
            if models['erosion']:
                erosion_prob, _ = predict_erosion(models['erosion'], temp_path)
                result['erosion_probability'] = erosion_prob
            
            results_summary.append(result)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            progress_bar.progress((idx + 1) / len(batch_files))
        
        # Display summary
        st.subheader("Batch Results Summary")
        import pandas as pd
        df_results = pd.DataFrame(results_summary)
        st.dataframe(df_results)
        
        # Download results
        csv = df_results.to_csv(index=False)
        st.download_button("Download Results CSV", csv, "batch_results.csv", "text/csv")

with tab3:
    st.header("About This Dashboard")
    
    st.markdown("""
    ### 🏛️ Ruins & Vegetation Analysis Dashboard
    
    This interactive dashboard integrates three AI models for comprehensive terrain analysis:
    
    #### 1. **Segmentation Model (DeepLabV3)**
    - Semantic segmentation of vegetation and ruins
    - Pixel-level classification
    - Coverage percentage calculation
    
    #### 2. **Detection Model (YOLOv5)**
    - Object detection and classification
    - Bounding box visualization
    - Confidence scores
    
    #### 3. **Erosion Prediction Model (XGBoost)**
    - Terrain erosion probability prediction
    - Risk level assessment
    - Feature-based analysis
    
    ### Features
    - ✅ Single image analysis
    - ✅ Batch processing
    - ✅ Unified visualization
    - ✅ Detailed metrics
    - ✅ Export results
    
    ### Model Performance
    - **Segmentation**: DeepLabV3 with custom classes
    - **Detection**: YOLOv5 (mAP: 2.11%, trained on artifacts)
    - **Erosion**: XGBoost (R²: 0.9985, RMSE: 0.0057)
    
    ### Usage
    1. Upload an image or select from dataset
    2. Configure model paths in sidebar (if needed)
    3. View integrated results
    4. Download results for further analysis
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Ruins & Vegetation Analysis Dashboard v1.0</p>", 
            unsafe_allow_html=True)

