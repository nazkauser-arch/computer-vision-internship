import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_URL = os.getenv("API_URL")
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(page_title="Helmet Detection App")

st.title("Helmet Detection App")
st.write("Upload an image to detect whether people are wearing helmets.")

uploaded_file = st.file_uploader(
    "Upload an image:",
    type=["jpg", "jpeg", "png"],
    max_upload_size=5
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image")

    if st.button("Detect"):

        try:
            with st.spinner("Detecting..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                data = {
                    "confidence": confidence
                }

                response = requests.post(
                    PREDICT_URL,
                    files=files,
                    data=data,
                    timeout=30
                )

            if response.status_code == 200:
                st.success("Image sent successfully to the API.")

            elif response.status_code == 400:
                detail = response.json().get(
                    "detail",
                    "Invalid request."
                )
                st.error(f"Error 400: {detail}")

            elif response.status_code == 413:
                detail = response.json().get(
                    "detail",
                    "File is too large. Maximum size is 5 MB."
                )
                st.error(f"Error 413: {detail}")

            else:
                detail = response.json().get(
                    "detail",
                    "Prediction failed."
                )
                st.error(
                    f"Error {response.status_code}: {detail}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "API is not available. Please make sure the FastAPI server is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "Request timed out. The API took too long to respond."
            )

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the prediction API."
            )