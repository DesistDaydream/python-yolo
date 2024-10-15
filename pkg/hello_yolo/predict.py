# detection 与 predict 有什么区别？
from ultralytics import YOLO

model = YOLO("models/yolo11x.pt")
list = model.predict(source="test_files/1.jpg", save=True)
