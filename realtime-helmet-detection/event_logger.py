import csv
import os
from datetime import datetime, timezone


OUTPUT_FILE = "output/tracking/events.csv"

logged_ids = set()


def log_event(tracking_id, class_name, confidence):
    if tracking_id in logged_ids:
        return

    file_exists = os.path.exists(OUTPUT_FILE)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "tracking_id",
                "class",
                "confidence"
            ])

        writer.writerow([
            timestamp,
            tracking_id,
            class_name,
            round(float(confidence), 2)
        ])

    logged_ids.add(tracking_id)