# detection 与 predict 有什么区别？
from ultralytics import YOLO

model = YOLO("models/yolo11x.pt")
results = model("test_files/bus.jpg")
results[0].show()
# for r in results:
#     print(f"Detected {len(r)} objects in image")
