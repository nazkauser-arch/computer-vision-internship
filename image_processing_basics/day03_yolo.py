from ultralytics import YOLO
import shutil
from moviepy import VideoFileClip

def load_model():
    model = YOLO("yolo26n.pt")
    return model

def detect_objects(model):
    results = model.predict(
        source = "images/input_yolo.jpg",
        conf = 0.25,
        save = True,
        project = "output",
        name = "input_yolo",
        verbose = False
    )
    return results

def print_detections(results, model):
    total_objects = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(f"Class ID: {class_id}")
            print(f"Object: {class_name}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Box: {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")
            print("\n")

            total_objects += 1

    print(f"Total objects detected: {total_objects}")

def count_objects(results, model, count={}):
    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if not class_name in count:
                count[class_name] = 1
            else:
                count[class_name] += 1

        if len(count) == 0:
            print("No objects detected")

    return count

def compare_confidence_levels(model):
    result_conf20 = model.predict(
        source = "images/input_yolo.jpg",
        conf = 0.20,
        save = True,
        project = "output",
        name = "conf_20",
        verbose = False
    )

    result_conf50 = model.predict(
        source = "images/input_yolo.jpg",
        conf = 0.50,
        save = True,
        project = "output",
        name = "conf_50",
        verbose = False
    )

    result_conf80 = model.predict(
        source = "images/input_yolo.jpg",
        conf = 0.80,
        save = True,
        project = "output",
        name = "conf_80",
        verbose = False
    )

    return result_conf20, result_conf50, result_conf80

def detect_selected_class(results, selected_class):
    results_selected_class = model.predict(
        source = "images/input_yolo.jpg",
        conf = 0.25,
        save = True,
        classes = [selected_class],
        project = "output",
        name = "filtered_class",
        verbose = False
    )

    return results_selected_class

def process_video(model):
    results_video = model.predict(
        source = "videos/input.mp4",
        conf = 0.40,
        stream = True,
        save = True,
        verbose = False,
        project = "output",
        name = "input_video"
    )
    
    for result in results_video:
        pass

    return results_video

# loading model
model = load_model()

# saving results --> image with objects detected
results = detect_objects(model)

# printing detections
print_detections(results, model)

# printing no of objects
print(count_objects(results, model, {}))

# confidence thresholds
compare_confidence_levels(model)

# detect selected object
detect_selected_class(results, 7)

# video
process_video(model)

# saving video as .mp4 file
video = VideoFileClip("runs/detect/output/input_video/input.avi")
video.write_videofile("output/day_03/yolo_result.mp4")
video.close()

# for conf = 0.25
shutil.copy("runs/detect/output/input_yolo/input_yolo.jpg", "output/day_03/yolo_result.jpg")

# for conf = 0.20
shutil.copy("runs/detect/output/conf_20/input_yolo.jpg", "output/day_03/conf_20.jpg")

# for conf = 0.50
shutil.copy("runs/detect/output/conf_50/input_yolo.jpg", "output/day_03/conf_50.jpg")

# for conf = 0.80
shutil.copy("runs/detect/output/conf_80/input_yolo.jpg", "output/day_03/conf_80.jpg")

# for filtered class
shutil.copy("runs/detect/output/filtered_class/input_yolo.jpg", "output/day_03/filtered_class.jpg")