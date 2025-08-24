# webcam_yolov8.py
import time
import cv2
from ultralytics import YOLO

# Load model (nano is fastest)
# model = YOLO("yolov8n.pt")  # or "yolov8s.pt" / "yolov8n-seg.pt" for segmentation
# model = YOLO("/Users/sameerkhan/Desktop/sameerkhan/data/cv/yolov8/yolov8n-pose.pt")
model = YOLO("/Users/sameerkhan/Desktop/sameerkhan/data/cv/experiments/yolov8_custom_acne/weights/best.pt")

# Try default camera; fallback to index 1 if needed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try a different index or check camera permissions.")

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Inference (conf adjusts sensitivity)
    results = model(frame, conf=0.5, verbose=False)

    # Annotated frame
    annotated = results[0].plot()

    # FPS overlay
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLOv8 Webcam (press q to quit, s to save)", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        cv2.imwrite(f"frame_{int(time.time())}.jpg", annotated)
        print("Saved frame.")

cap.release()
cv2.destroyAllWindows()