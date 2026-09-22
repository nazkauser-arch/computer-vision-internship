import streamlit as st
from PIL import Image

st.set_page_config(page_title="Helmet Detection App")

st.title("Helmet Detection App")
st.write("Upload image to detect whether people are wearing helmets.")

uploaded_file = st.file_uploader(
    "Upload an image:",
    type = ["jpg", "jpeg", "png"],
    max_upload_size = 5
)

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Original image")
