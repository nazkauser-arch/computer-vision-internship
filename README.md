# Computer Vision Internship

This repository contains the projects, experiments, and implementations completed during my **Computer Vision Internship at Devsinc**.

The internship focuses on **computer vision, object detection, model evaluation, API development, frontend integration, Dockerization and real-time object tracking**.

## Projects

### 1. Helmet Detection

A YOLO-based helmet detection system that identifies whether a person is wearing a helmet.

**Classes:**

* `helmet`
* `no_helmet`

The model was trained using a Roboflow dataset containing annotated traffic images.

**Main technologies:**

* Python
* YOLO / Ultralytics
* OpenCV
* Roboflow

### 2. Model Evaluation & Error Analysis

The trained helmet detection model was evaluated using validation data.

The evaluation included:

* Precision
* Recall
* mAP
* Detection results

Model errors were categorized into:

| Error Type        | Description                                               |
| ----------------- | --------------------------------------------------------- |
| False Positive    | The model detected an invalid object                      |
| False Negative    | A valid helmet/no-helmet case was missed                  |
| Wrong Class       | The object was detected but assigned the wrong class      |
| Poor Bounding Box | The class was correct but the bounding box was inaccurate |

### 3. Helmet Detection API

A FastAPI backend was developed to serve the trained YOLO model.

The API accepts an uploaded image and returns detection results.

**Endpoints:**

```text
GET /health
POST /predict
```

The `/predict` endpoint returns:

* Detected class
* Confidence score
* Bounding box coordinates
* Helmet count
* No-helmet count
* Processing time

### 4. Helmet Detection Frontend

A Streamlit frontend was developed to interact with the detection API.

Features include:

* Image upload
* Detection request
* Helmet/no-helmet counts
* Detection confidence
* Bounding boxes
* Processing time
* API error handling

### 5. Dockerization

The helmet detection application was containerized using Docker and Docker Compose.

```text
Frontend
   ↓
FastAPI Backend
   ↓
YOLO Helmet Detection Model
```

### 6. Real-Time Helmet Tracking

A real-time tracking implementation was developed to process video using YOLO tracking.

The system:

* Detects helmets and no-helmet cases
* Assigns tracking IDs
* Maintains IDs across frames
* Avoids repeatedly counting the same tracked object
* Displays live detection counts
* Calculates FPS
* Processes video input
* Saves processed output video

## Technologies Used

* **Python**
* **Ultralytics YOLO**
* **OpenCV**
* **FastAPI**
* **Streamlit**
* **Docker**
* **Docker Compose**
* **Roboflow**
* **Git & GitHub**

## Learning Outcomes

During this internship at **Devsinc**, I have worked with:

* Object detection using YOLO
* Dataset preparation and annotation
* Model training and evaluation
* Computer vision with OpenCV
* Detection error analysis
* REST API development with FastAPI
* Frontend development with Streamlit
* Backend/frontend integration
* Docker and Docker Compose
* Real-time object tracking
* Git and GitHub workflows

## Author

**Kauser Naz**

Computer Vision Intern at **Devsinc**
