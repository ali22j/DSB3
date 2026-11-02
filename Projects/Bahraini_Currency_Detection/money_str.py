
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Bahrain Currency Detector")

@st.cache_resource
def load_model(path):
    return YOLO(path)

def main():
    st.title("Bahrain Currency Object Detector")
    st.write("Upload an image to detect Bahraini currency notes and coins.")

    model_path = st.sidebar.text_input("Model Path", "best.pt")
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file)
        results = model.predict(image, conf=confidence)
        result = results[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Results")
            st.image(result.plot()[:, :, ::-1], use_container_width=True)

        st.subheader("Detection Summary")

        if len(result.boxes):
            for i, box in enumerate(result.boxes):
                name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                st.write(f"**Object {i+1}:** {name} ({conf:.2f})")
        else:
            st.info("No objects detected.")

if __name__ == "__main__":
    main()

