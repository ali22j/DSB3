import json
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from search_core import load_model, preprocess_query, embed, search

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_index():
    with open("index/config.json") as f:
        config = json.load(f)
    return (np.load("index/embeddings.npy"), np.load("index/gallery_images.npy"),
            config["threshold"], config["top_k"])

model = get_model()
embeddings, gallery_images, threshold, top_k = get_index()

def show_results(image):
    small = preprocess_query(image)
    if small is None:
        st.info("Nothing drawn yet.")
        return

    col1, col2 = st.columns(2)
    col1.image(image, caption="Your input", width=200)
    col2.image(small, caption="What the model sees (28 x 28)", width=200)

    indices, scores = search(embed(model, small.reshape(1, 28, 28)), embeddings, top_k)
    indices, scores = indices[0], scores[0]

    if scores[0] < threshold:
        st.warning("Nothing similar found.")
        return

    st.subheader("Top 25 similar doodles")
    for n in range(min(25, len(indices))):
        if n % 5 == 0:
            columns = st.columns(5)
        columns[n % 5].image(gallery_images[indices[n]], caption=str(round(float(scores[n]), 2)), width=100)

st.title("DoodleDex")
st.write("Draw something, upload a drawing, or show a paper drawing to the camera.")
tab_draw, tab_upload, tab_camera = st.tabs(["Draw", "Upload", "Camera"])

with tab_draw:
    canvas = st_canvas(stroke_width=8, stroke_color="black", background_color="white",
                       height=280, width=280, drawing_mode="freedraw",
                       update_streamlit=True, key="canvas")
    if canvas.image_data is not None:
        show_results(Image.fromarray(canvas.image_data.astype("uint8")))

with tab_upload:
    file = st.file_uploader("Upload a drawing", type=["png", "jpg", "jpeg"])
    if file:
        show_results(Image.open(file).convert("RGB"))

with tab_camera:
    photo = st.camera_input("Take a photo of a drawing on paper")
    if photo:
        show_results(Image.open(photo).convert("RGB"))

st.caption("Doodles: Quick, Draw! dataset by Google, Inc., CC BY 4.0.")

