from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from ultralytics import YOLO
import time

app = FastAPI(title="Helmet Detection API")
model = YOLO("model/best.pt")

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model" : "helmet-detector" 
    }


async def validate_upload(file: UploadFile):
    allowed_extensions = [".jpg", ".jpeg", ".png"]

    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only JPG, JPEG and PNG are allowed."
        )

    max_file_size = 5 * 1024 * 1024
    contents = bytearray()

    while chunk := await file.read(1024 * 1024):
        contents.extend(chunk)

        if len(contents) > max_file_size:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 5 MB."
            )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    return contents

def decode_image(contents):
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    return image

def run_prediction(image):
    try:
        start_time = time.perf_counter()

        results = model.predict(
            source=image,
            conf=0.50,
            verbose=False
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return results, processing_time_ms

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

def format_detections(results):
    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        detection = {
            "class": class_name,
            "confidence": confidence,
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }

        detections.append(detection)

    return detections

def count_classes(detections):
    counts = {
        "helmet": 0,
        "no_helmet": 0
    }

    for detection in detections:
        class_name = detection["class"]
        counts[class_name] += 1

    return counts

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await validate_upload(file)
    image = decode_image(contents)
    results, processing_time_ms = run_prediction(image)
    detections = format_detections(results)
    counts = count_classes(detections)

    return {
        "message": "Image uploaded and decoded successfully",
        "filename": file.filename,
        "content_type": file.content_type,
        "width": image.shape[1],
        "height": image.shape[0],
        "detections": detections,
        "counts": counts,
        "processing_time_ms": round(processing_time_ms, 2)
    }