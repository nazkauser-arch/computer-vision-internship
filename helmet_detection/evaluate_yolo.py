from ultralytics import YOLO

model = YOLO("runs/detect/helmet_detector/weights/best.pt")

metrics = model.val(
    data="helmet-dataset/data.yaml",
    split="val"
)