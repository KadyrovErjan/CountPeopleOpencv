import cv2
import time
from ultralytics import YOLO

VIDEO_SOURCE = 0  # локально: 0, на AWS лучше RTSP или файл

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Camera / video source not found")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("YOLO started...")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

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
            label = model.names[cls]

            if label == "person":
                person_count += 1

    fps = 1 / (time.time() - start_time + 1e-6)

    print(f"FPS: {fps:.1f} | Persons: {person_count}")

cap.release()
