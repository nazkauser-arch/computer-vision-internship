import streamlit as st
from PIL import Image, ImageDraw
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

                result = response.json()

                detections = result["detections"]
                counts = result["counts"]
                processing_time = result["processing_time_ms"]

                result_image = image.copy()
                draw = ImageDraw.Draw(result_image)

                for detection in detections:

                    class_name = detection["class"]
                    confidence_value = detection["confidence"]
                    box = detection["box"]

                    x1 = box["x1"]
                    y1 = box["y1"]
                    x2 = box["x2"]
                    y2 = box["y2"]

                    if class_name == "helmet":
                        draw.rectangle(
                            [x1, y1, x2, y2],
                            outline="green",
                            width=3
                        )

                        draw.text(
                            (x1, y1),
                            f"{class_name} {confidence_value:.2f}",
                            fill="green"
                        )
                    elif class_name == "no_helmet":
                        draw.rectangle(
                            [x1, y1, x2, y2],
                            outline="red",
                            width=3
                        )

                        draw.text(
                            (x1, y1),
                            f"{class_name} {confidence_value:.2f}",
                            fill="red"
                        )

                st.subheader("Detection Result")
                st.image(result_image)

                st.subheader("Detection Counts")

                st.write(f"Helmet: {counts['helmet']}")
                st.write(f"No Helmet: {counts['no_helmet']}")
                st.write(f"Processing Time: {processing_time:.2f} ms")

                st.subheader("Detection Details")

                if detections:
                    table_data = []

                    for i, detection in enumerate(detections, start=1):

                        table_data.append({
                            "Detection": i,
                            "Class": detection["class"],
                            "Confidence": round(
                                detection["confidence"], 2
                                )
                            })

                    st.table(table_data)

                else:
                    st.write("No detections found.")

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