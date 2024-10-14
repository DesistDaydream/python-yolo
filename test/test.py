from ultralytics import YOLO
model = YOLO("models/yolo11n.pt")
results = model("test_files/1.jpg")