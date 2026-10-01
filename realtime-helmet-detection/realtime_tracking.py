from ultralytics import YOLO
import cv2
import time
from event_logger import log_event

model = YOLO("model/best.pt")

camera = cv2.VideoCapture("videos/helmet_test.mp4")

if not camera.isOpened():
    print("Error: Could not open video")
    exit()

screen_width = 1280
screen_height = 720

cv2.namedWindow("Helmet Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Helmet Tracking", screen_width, screen_height)

frame_helmets = 0
frame_no_helmets = 0
seen_track_ids = set()
fps = 0

frame_count = 0
start_time = time.time()

no_helmet_counts = {}
confirmed_events = set()

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            tracker="custom_bytetrack.yaml",
            conf=0.50,
            imgsz = 640,
            verbose=False
        )

        frame_helmets = 0
        frame_no_helmets = 0

        if not results:
            continue

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            box = [
                int(value)
                for value in boxes.xyxy[i].cpu().tolist()
            ]

            x1, y1, x2, y2 = box

            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i].item())

            tracking_id = None

            if boxes.id is not None:
                tracking_id = int(boxes.id[i].item())

            if tracking_id is not None:
                seen_track_ids.add(tracking_id)

                if class_name == "no_helmet":
                    no_helmet_counts[tracking_id] = (
                        no_helmet_counts.get(tracking_id, 0) + 1
                    )

                    if no_helmet_counts[tracking_id] >= 5:
                        if tracking_id not in confirmed_events:
                            confirmed_events.add(tracking_id)

                            print(
                                f"CONFIRMED NO-HELMET EVENT: ID {tracking_id}"
                            )
                            log_event(tracking_id, "no_helmet", confidence)

                elif class_name == "helmet":
                    no_helmet_counts[tracking_id] = 0

            if class_name == "helmet":
                color = (0, 255, 0)
                frame_helmets += 1

            elif class_name == "no_helmet":
                color = (0, 0, 255)
                frame_no_helmets += 1

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                4
            )

            label = (
                f"ID: {tracking_id} | "
                f"{class_name} | "
                f"Conf: {confidence * 100:.0f}%"
            )

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

        current_helmets = frame_helmets
        current_no_helmets = frame_no_helmets

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time

        cv2.putText(
            frame,
            f"Current helmets: {current_helmets}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4
        )

        cv2.putText(
            frame,
            f"Current no-helmets: {current_no_helmets}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

        cv2.putText(
            frame,
            f"Unique people seen: {len(seen_track_ids)}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4
        )

        height, width = frame.shape[:2]

        scale = min(
            screen_width / width,
            screen_height / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height)
        )

        cv2.imshow("Helmet Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Total confirmed no helmet events: {len(confirmed_events)}")

finally:
    end_time = time.time()
    processing_duration = end_time - start_time
    print(f"Processing duration: {processing_duration:.2f} seconds")

    camera.release()
    cv2.destroyAllWindows()