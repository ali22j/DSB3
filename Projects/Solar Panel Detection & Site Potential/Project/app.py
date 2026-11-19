import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.title("Solar Panel Segmentation & Counter")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.15, 0.05)
file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    image = np.array(Image.open(file).convert("RGB"))
    result = model.predict(image, conf=conf, retina_masks=True, verbose=False)[0]
    plotted = image.copy()
    count = 0

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        count = len(masks)

        for mask in masks:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(plotted, contours, -1, (0, 100, 255), 2)

    c1, c2 = st.columns(2)
    c1.image(image, caption="Original Image", use_container_width=True)
    c2.image(plotted, caption="Solar Panel Detection", use_container_width=True)