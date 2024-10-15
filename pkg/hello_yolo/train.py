from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")
results = model.train(data="models/custom.yaml", epochs=3)