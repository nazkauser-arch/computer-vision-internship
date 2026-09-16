import os
import random
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = "helmet-dataset"
CLASS_NAMES = ["helmet", "no_helmet"]
SPLITS = ["train", "valid", "test"]

def get_dataset_statistics():
    total_images = 0
    split_counts = {}
    helmet_labels = 0
    no_helmet_labels = 0
    images_without_labels = 0

    for split in SPLITS:
        image_dir = os.path.join(DATASET_PATH, split, "images")
        label_dir = os.path.join(DATASET_PATH, split, "labels")
        image_files = [
            file for file in os.listdir(image_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        split_counts[split] = len(image_files)
        total_images += len(image_files)

        for image_file in image_files:
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                images_without_labels += 1
                continue

            with open(label_path, "r") as file:
                labels = file.readlines()

            for label in labels:
                parts = label.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                if class_id == 0:
                    helmet_labels += 1

                elif class_id == 1:
                    no_helmet_labels += 1

    print("\nDataset Statistics")
    print("------------------")
    print(f"Total images: {total_images}")
    print(f"Training images: {split_counts['train']}")
    print(f"Validation images: {split_counts['valid']}")
    print(f"Test images: {split_counts['test']}")
    print(f"Helmet labels: {helmet_labels}")
    print(f"No-helmet labels: {no_helmet_labels}")
    print(f"Images without labels: {images_without_labels}")


def display_random_images():
    images_to_show = []

    for split in SPLITS:
        image_dir = os.path.join(DATASET_PATH, split, "images")
        label_dir = os.path.join(DATASET_PATH, split, "labels")

        image_files = [
            file for file in os.listdir(image_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        selected_images = random.sample(
            image_files,
            min(5, len(image_files))
        )

        for image_file in selected_images:
            image_path = os.path.join(image_dir, image_file)

            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]

            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    labels = file.readlines()

                for label in labels:
                    parts = label.strip().split()

                    if not parts:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    x1 = int((x_center - box_width / 2) * width)
                    y1 = int((y_center - box_height / 2) * height)
                    x2 = int((x_center + box_width / 2) * width)
                    y2 = int((y_center + box_height / 2) * height)

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        image,
                        CLASS_NAMES[class_id],
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            images_to_show.append((split, image_file, image))

    for split, image_file, image in images_to_show:
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"{split}: {image_file}")
        plt.axis("off")
        plt.show()


get_dataset_statistics()
display_random_images()