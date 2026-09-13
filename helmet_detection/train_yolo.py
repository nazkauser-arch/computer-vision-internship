from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data="helmet-dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    project="helmet_detector",
    name="helmet_detector"
)