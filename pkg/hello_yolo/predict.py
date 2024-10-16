from ultralytics import YOLO
from ultralytics.engine.model import Model

model = YOLO("models/yolo11x.pt")

# predict() 方法有一个别名，__call__() 中的逻辑是直接  return self.predict()
# 所以 model.predict() 可以简写为 model()。比如：
# from ultralytics.engine.results import Results
# results: Results = model("test_files/1.jpg")
results = model.predict(source="test_files/1.jpg", save=True)
for result in results:
    print(result.boxes)
