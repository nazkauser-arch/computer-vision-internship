from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data = "helmet-dataset/data.yaml",
    epochs = 30,
    imgsz = 640,
    batch = 8,
    name = "helmet_detector"
)

# version 2
model.train(
    data = "helmet-dataset/data.yaml",
    epochs = 30,
    imgsz = 640,
    batch = 8,
    name = "helmet_detector_v2"
)