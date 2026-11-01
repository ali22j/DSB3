import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Webcam Age Predictor", layout="centered"
)

st.title("Real-Time Age Prediction App")
st.write(
    "Capture a photo using your webcam to estimate your age using the trained"
    " CNN model."
)


# Load the trained Keras model (cached for performance)
@st.cache_resource
def load_age_model():
  return tf.keras.models.load_model("utkface_age_regression_model.keras")


try:
  model = load_age_model()
except Exception as e:
  st.error(
      "Could not find 'utkface_age_regression_model.keras'. Please run"
      f" `train.py` first. Error: {e}"
  )
  st.stop()

IMG_SIZE = 128

# Streamlit webcam widget
captured_image = st.camera_input("Take a picture")

if captured_image is not None:
  # Open image via PIL
  image = Image.open(captured_image)
  st.image(image, caption="Captured Image", use_container_width=True)

  # Preprocess image to match training parameters
  image_resized = image.resize((IMG_SIZE, IMG_SIZE))
  img_array = np.array(image_resized) / 255.0
  img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

  # Predict age
  prediction = model.predict(img_array)
  predicted_age = prediction[0][0]

  # Display result
  st.success(f"### Estimated Age: {predicted_age:.1f} years old")