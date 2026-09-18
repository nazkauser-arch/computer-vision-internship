from ultralytics import YOLO

model = YOLO("runs/detect/helmet_detector_v2/weights/best.pt")

metrics = model.val(
    data="helmet-dataset/data.yaml",
    split="val"
)

# conf = 0.25
model.predict(
    source = "helmet-dataset/valid/images",
    conf = 0.25,
    save = True,
    project = "predictions",
    name = "validation_review_conf25"
)

# conf = 0.50
model.predict(
    source = "helmet-dataset/valid/images",
    conf = 0.50,
    save = True,
    project = "predictions",
    name = "validation_review_conf50"
)

# conf = 0.75
model.predict(
    source = "helmet-dataset/valid/images",
    conf = 0.75,
    save = True,
    project = "predictions",
    name = "validation_review_conf75"
)