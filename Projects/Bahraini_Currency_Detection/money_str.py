import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Bahraini Currency Detector")
st.title("Bahraini Currency Detector")
st.write("Detect Bahraini currency using YOLO!")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def predict(image):
    results = model(image)[0]
    st.image(cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB),
             caption="Prediction Result")

    boxes = results.boxes
    if len(boxes):
        st.success(f"Detected {len(boxes)} currency object(s)!")
        for box in boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- **{name}** (Confidence: {conf:.2f})")
    else:
        st.warning("No currency detected. Try another image.")

mode = st.selectbox("Choose Input Mode", ["Live Webcam", "Upload Image"])

if mode == "Live Webcam":
    image = st.camera_input("Take a picture of the currency")
    if image:
        predict(Image.open(image))

else:
    image = st.file_uploader("Upload currency image", type=["jpg", "jpeg", "png"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Uploaded Image")
        if st.button("Predict Currency"):
            with st.spinner("Analyzing..."):
                predict(img)
