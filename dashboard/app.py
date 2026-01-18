import streamlit as st
import pandas as pd
from PIL import Image
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI-Based Archaeological Monitoring", layout="wide")

st.title("AI-Based Archaeological Site Monitoring System")
st.markdown("### Segmentation • Object Detection • Terrain Erosion Prediction")

# -----------------------------
# PATHS (UPDATED FOR YOUR SYSTEM)
# -----------------------------
INPUT_DIR = r"D:\Infosys Project\yolo\images\val"

YOLO_DIR = r"D:\Infosys Project\yolo\runs\detect\predict2"
YOLO_LABELS_DIR = r"D:\Infosys Project\yolo\runs\detect\predict2\labels"

YOLO_RESULTS_CSV = r"D:\Infosys Project\yolo\runs\detect\train3\results.csv"

EROSION_DATA_PATH = r"D:\Infosys Project\Erosion\data\erosion_dataset_augmented.csv"

# If you have only one class in YOLO
CLASS_NAMES = {
    0: "Archaeological_Structure"
}

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Select Analysis")
option = st.sidebar.selectbox(
    "Choose Module",
    ["YOLO Object Detection", "Terrain Erosion Prediction"]
)

# -----------------------------
# YOLO OBJECT DETECTION MODULE
# -----------------------------
if option == "YOLO Object Detection":
    st.subheader("Object Detection using YOLO")

    # Read images safely
    if not os.path.exists(INPUT_DIR):
        st.error("Input folder not found!")
        st.code(INPUT_DIR)
        st.stop()

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(images) == 0:
        st.warning("No images found in input folder.")
        st.code(INPUT_DIR)
        st.stop()

    selected_image = st.selectbox("Select Image", images)

    col1, col2 = st.columns(2)

    # ✅ Original Image
    with col1:
        st.markdown("### Original Image")
        original_path = os.path.join(INPUT_DIR, selected_image)
        img = Image.open(original_path)
        st.image(img, use_container_width=True)

    # ✅ YOLO Detection Output Image
    with col2:
        st.markdown("### YOLO Detection Output")

        # YOLO output filename is same as input name (but extension may differ)
        detected_path = os.path.join(YOLO_DIR, selected_image)

        # If exact file not found, try switching extension
        if not os.path.exists(detected_path):
            base_name = os.path.splitext(selected_image)[0]
            alt1 = os.path.join(YOLO_DIR, base_name + ".jpg")
            alt2 = os.path.join(YOLO_DIR, base_name + ".png")

            if os.path.exists(alt1):
                detected_path = alt1
            elif os.path.exists(alt2):
                detected_path = alt2

        if os.path.exists(detected_path):
            det_img = Image.open(detected_path)
            st.image(det_img, use_container_width=True)
        else:
            st.error("Detection result image not found")
            st.write("Expected file at:")
            st.code(detected_path)

    # -----------------------------
    # DETECTION COUNT + CONFIDENCE
    # -----------------------------
    st.markdown("---")
    st.subheader("Detection Summary (Count + Confidence)")

    base_name = os.path.splitext(selected_image)[0]
    label_path = os.path.join(YOLO_LABELS_DIR, base_name + ".txt")

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        detections = []
        for line in lines:
            parts = line.split()

            # Expected YOLO format: class x y w h conf
            if len(parts) >= 6:
                cls_id = int(float(parts[0]))
                conf = float(parts[5])
                cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
                detections.append((cls_id, cls_name, conf))

        st.success(f"✅ Total Objects Detected: {len(detections)}")

        if len(detections) > 0:
            df_det = pd.DataFrame(detections, columns=["Class_ID", "Class_Name", "Confidence"])
            st.dataframe(df_det)

            st.write("✅ **Average Confidence:**", round(df_det["Confidence"].mean(), 3))
            st.write("✅ **Max Confidence:**", round(df_det["Confidence"].max(), 3))
            st.write("✅ **Min Confidence:**", round(df_det["Confidence"].min(), 3))
        else:
            st.info("No objects detected in this image.")
    else:
        st.warning("Label file not found for selected image.")
        st.write("Expected label file at:")
        st.code(label_path)
        st.info("Make sure you predicted with: save_txt=True save_conf=True")

    # -----------------------------
    # SHOW YOLO TRAINING METRICS
    # -----------------------------
    st.markdown("---")
    st.subheader("YOLO Training Metrics (results.csv)")

    if os.path.exists(YOLO_RESULTS_CSV):
        results_df = pd.read_csv(YOLO_RESULTS_CSV)

        # show last 5 epochs
        st.dataframe(results_df.tail(5))

        # Show final epoch quick summary
        last_row = results_df.iloc[-1]

        st.markdown("### Final Epoch Summary")
        st.write("✅ Final Epoch:", int(last_row["epoch"]) if "epoch" in results_df.columns else "N/A")

        # Columns differ depending on version, so we check safely
        for col in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
            if col in results_df.columns:
                st.write(f"✅ {col}: {round(float(last_row[col]), 3)}")

    else:
        st.warning("results.csv file not found!")
        st.write("Expected file at:")
        st.code(YOLO_RESULTS_CSV)

# -----------------------------
# TERRAIN EROSION MODULE
# -----------------------------
elif option == "Terrain Erosion Prediction":
    st.subheader("Terrain Erosion Prediction Dashboard")

    if not os.path.exists(EROSION_DATA_PATH):
        st.error("Erosion dataset CSV not found!")
        st.code(EROSION_DATA_PATH)
        st.stop()

    df = pd.read_csv(EROSION_DATA_PATH)

    # -----------------------------
    # TOP SUMMARY CARDS
    # -----------------------------
    st.markdown("### Dataset Overview")

    total_rows = len(df)
    total_features = df.shape[1]

    stable_count = int((df["Erosion_Label"] == 0).sum()) if "Erosion_Label" in df.columns else 0
    prone_count = int((df["Erosion_Label"] == 1).sum()) if "Erosion_Label" in df.columns else 0

    avg_score = float(df["Erosion_Score"].mean()) if "Erosion_Score" in df.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", total_rows)
    c2.metric("Total Columns", total_features)
    c3.metric("Stable Areas (0)", stable_count)
    c4.metric("Erosion-Prone Areas (1)", prone_count)

    if avg_score is not None:
        st.info(f"✅ Average Erosion Severity Score: **{avg_score:.3f}**")

    st.markdown("---")

    # -----------------------------
    # FILTERS
    # -----------------------------
    st.markdown("### Filters")

    colf1, colf2, colf3 = st.columns(3)

    vegetation_filter = None
    terrain_filter = None
    label_filter = None

    if "Vegetation" in df.columns:
        with colf1:
            vegetation_options = ["All"] + sorted(df["Vegetation"].astype(str).unique().tolist())
            vegetation_filter = st.selectbox("Vegetation Level", vegetation_options)

    if "Terrain_Type" in df.columns:
        with colf2:
            terrain_options = ["All"] + sorted(df["Terrain_Type"].astype(str).unique().tolist())
            terrain_filter = st.selectbox("Terrain Type", terrain_options)

    if "Erosion_Label" in df.columns:
        with colf3:
            label_filter = st.selectbox("Erosion Risk Label", ["All", "0 (Stable)", "1 (Erosion-Prone)"])

    # Apply filters
    filtered_df = df.copy()

    if vegetation_filter and vegetation_filter != "All":
        filtered_df = filtered_df[filtered_df["Vegetation"].astype(str) == vegetation_filter]

    if terrain_filter and terrain_filter != "All":
        filtered_df = filtered_df[filtered_df["Terrain_Type"].astype(str) == terrain_filter]

    if label_filter and label_filter != "All":
        if "0" in label_filter:
            filtered_df = filtered_df[filtered_df["Erosion_Label"] == 0]
        else:
            filtered_df = filtered_df[filtered_df["Erosion_Label"] == 1]

    st.markdown("---")

    # -----------------------------
    # TABLE VIEW
    # -----------------------------
    st.markdown("### Filtered Dataset View")
    st.write(f"Showing **{len(filtered_df)}** records after filtering.")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # VISUAL SUMMARY (CHARTS)
    # -----------------------------
    st.markdown("### Visual Summary")

    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Erosion Label Distribution
    if "Erosion_Label" in df.columns:
        with chart_col1:
            st.markdown("#### Erosion Risk Distribution")
            label_counts = df["Erosion_Label"].value_counts().sort_index()
            st.bar_chart(label_counts)

    # Chart 2: Average Erosion Score by Terrain Type
    if "Erosion_Score" in df.columns and "Terrain_Type" in df.columns:
        with chart_col2:
            st.markdown("#### Avg Erosion Score by Terrain Type")
            avg_by_terrain = df.groupby("Terrain_Type")["Erosion_Score"].mean().sort_values(ascending=False)
            st.bar_chart(avg_by_terrain)

    st.markdown("---")

    # -----------------------------
    # INTERACTIVE PREDICTION DEMO (OFFLINE)
    # -----------------------------
    st.markdown("### Erosion Prediction Demo (Manual Input)")

    p1, p2, p3, p4 = st.columns(4)
    slope = p1.slider("Slope", 0, 50, 25)
    elevation = p2.slider("Elevation", 0, 4000, 1000)

    veg_value = p3.selectbox("Vegetation", ["Low", "Medium", "High"])
    terrain_value = p4.selectbox("Terrain Type", ["Rocky", "Sandy", "Plateau", "Plain", "Rugged"])

    if st.button("Show Example Prediction"):
        # simple rule-based demo (offline visualization)
        severity = min(0.99, (slope / 50) * 0.7 + (0.2 if veg_value == "Low" else 0.1 if veg_value == "Medium" else 0.05))
        risk = 1 if severity > 0.55 else 0

        st.success("✅ Prediction Generated")
        st.write(f"**Predicted Erosion Risk Label:** `{risk}` ({'Erosion-Prone' if risk==1 else 'Stable'})")
        st.write(f"**Predicted Severity Score:** `{severity:.3f}`")

    st.info("This tab visualizes the erosion dataset and demonstrates prediction")
