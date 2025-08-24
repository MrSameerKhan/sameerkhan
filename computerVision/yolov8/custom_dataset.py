from ultralytics import YOLO

# Start with pretrained weights (recommended)
model = YOLO("/Users/sameerkhan/Desktop/sameerkhan/data/cv/yolov8/yolov8n.pt")

# Train
model.train(
    data="/Users/sameerkhan/Desktop/sameerkhan/data/cv/yolov8/custom dataset acne/data.yaml",
    epochs=5,
    batch=16,
    project="/Users/sameerkhan/Desktop/sameerkhan/data/cv/experiments",
    name="yolov8_custom_acne"
)