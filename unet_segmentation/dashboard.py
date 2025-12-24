import streamlit as st
import pandas as pd
import joblib
from PIL import Image
model = joblib.load("erosion_rf_model.pkl")


st.title("AI Archaeological Site Mapping Dashboard")
st.write("Predict erosion risk and visualize segmentation/detection results.")

st.header("Terrain Erosion Prediction")

uploaded_file = st.file_uploader("Upload terrain CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Input Terrain Data:")
    st.dataframe(data)

    prediction = model.predict(data)
    st.write(f"Predicted Erosion Risk: **{prediction[0]}**")

st.header("Segmentation Result (Placeholder)")
seg_image = Image.open("segmentation_sample.png")  
st.image(seg_image, caption="Segmentation of ruins/vegetation", use_column_width=True)

st.header("Artifact Detection Result (Placeholder)")
det_image = Image.open("detection_sample.png")  
st.image(det_image, caption="Detected artifacts", use_column_width=True)
