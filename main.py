import cv2
import time
import threading
from fastapi import FastAPI
from ultralytics import YOLO

VIDEO_SOURCE = "people.mp4"

app = FastAPI()

stats = {
    "fps": 0.0,
    "persons": 0,
    "status": "starting"
}

def yolo_worker():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        stats["status"] = "video_not_found"
        return

    stats["status"] = "running"

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(
            frame,
            conf=0.3,
            imgsz=416,
            device="cpu",
            verbose=False
        )

        boxes = results[0].boxes
        person_count = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    person_count += 1

        fps = 1 / (time.time() - start_time + 1e-6)

        stats["fps"] = round(fps, 2)
        stats["persons"] = person_count

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=yolo_worker, daemon=True)
    thread.start()

@app.get("/")
def root():
    return {
        "service": "YOLOv8 People Counter",
        "status": stats["status"]
    }

@app.get("/stats")
def get_stats():
    return stats