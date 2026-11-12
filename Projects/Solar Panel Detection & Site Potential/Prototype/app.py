import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("Solar Panel Segmentation & Counter")

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model("best.pt")
conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
file = st.file_uploader("upload img", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    result = model.predict(np.array(img), conf=conf)[0]
    plotted = result.plot()[:, :, ::-1]  # BGR -> RGB
    count = 0 if result.masks is None else len(result.masks)

    c1, c2 = st.columns(2)
    c1.image(img, caption="original img", use_container_width=True)
    c2.image(plotted, caption="results", use_container_width=True)
    st.metric("number of solar panels", count)