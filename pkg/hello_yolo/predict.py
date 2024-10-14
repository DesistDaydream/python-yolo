from ultralytics import YOLO
model = YOLO("models/yolo11n.pt")
list = model.predict(source="test_files/1.jpg", save=True)
