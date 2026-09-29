from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw


model = YOLO("model/best.pt")

camera = cv2.VideoCapture("videos/helmet_test.mp4")

if not camera.isOpened():
    print("Error: Could not open video")
    exit()


screen_width = 1280
screen_height = 720

cv2.namedWindow("Helmet Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Helmet Tracking", screen_width, screen_height)


try:
    while True:
        ret, frame = camera.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="custom_bytetrack.yaml",
            conf=0.10,
            verbose=False
        )

        if not results:
            continue

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().tolist()
            box = [int(value) for value in box]
            box = [int(value) for value in boxes.xyxy[i].cpu().tolist()]

            x1, y1, x2, y2 = box

            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i].item())
            tracking_id = None

            if boxes.id is not None:
                tracking_id = int(boxes.id[i].item())

            if class_name == "helmet":
                color = (0, 255, 0)
            elif class_name == "no_helmet":
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                4
            )

            label = f"{class_name} | Conf: {confidence:.2f} | ID: {tracking_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                7
            )

            print(
                f"Box: {box}, "
                f"Class ID: {class_id}, "
                f"Class Name: {class_name}, "
                f"Confidence: {confidence:.2f}, "
                f"Tracking ID: {tracking_id}"
            )

        annotated_frame = frame
        height, width = annotated_frame.shape[:2]

        scale = min(
            screen_width / width,
            screen_height / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        annotated_frame = cv2.resize(
            annotated_frame,
            (new_width, new_height)
        )

        cv2.imshow("Helmet Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()