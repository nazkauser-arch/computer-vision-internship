from ultralytics import YOLO

model = YOLO("runs/detect/helmet_detector/weights/best.pt")

# for conf = 0.25
model.predict(
    source="helmet-dataset/test/images",
    conf = 0.25,
    save = True,
    project = "predictions",
    name = "helmet_test_25"
)

# for conf = 0.50
model.predict(
    source = "helmet-dataset/test/images",
    conf = 0.50,
    save = True,
    project = "predictions",
    name = "helmet_test_50"
)

# for conf = 0.75
model.predict(
    source = "helmet-dataset/test/images",
    conf = 0.75,
    save = True,
    project = "predictions",
    name = "helmet_test_75"
)