from ultralytics import YOLO

# 从头创建一个新的 YOLO 模型。
# TODO: 这文件哪来的？下面不是加载模型了么？
model = YOLO("yolo11n.yaml")

# 加载预训练的YOLO模型（推荐用于训练）
# 找不到的话会去 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt 下载到项目根目录
model = YOLO("models/yolo11n.pt")

# 使用 coco8.yaml 数据集训练模型 3 个周期
results = model.train(data="coco8.yaml", epochs=3)

# 评估模型在验证集上的性能
results = model.val()

# 使用模型对图像执行对象检测
results = model("https://ultralytics.com/images/bus.jpg")

# 将模型导出为 ONNX 格式
success = model.export(format="onnx")


