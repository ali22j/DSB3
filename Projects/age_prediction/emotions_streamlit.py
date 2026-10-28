import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="centered"
)

st.title("Facial Emotion Recognition App")
st.write("Take a photo using your webcam to detect facial emotions.")

@st.cache_resource
def load_emotion_model():
    model_path = Path(__file__).resolve().parent / "emotion_model.keras"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_emotion_model()

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_emotion(image: Image.Image):
    img_gray = image.convert('L')
    img_resized = img_gray.resize((48, 48))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=(0, -1))
    predictions = model.predict(img_tensor)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    return EMOTION_LABELS[predicted_class_idx], confidence, predictions[0]

camera_file = st.camera_input("Take a picture")

if camera_file is not None:
    uploaded_image = Image.open(camera_file)
    with st.spinner("Analyzing emotion..."):
        label, conf, all_probs = predict_emotion(uploaded_image)
    st.success(f"**Predicted Emotion:** {label} ({conf*100:.2f}% confidence)")
    st.write("### Prediction Probabilities")
    for emotion, prob in zip(EMOTION_LABELS, all_probs):
        st.progress(float(prob), text=f"{emotion}: {prob*100:.2f}%")
