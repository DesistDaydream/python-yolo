from ultralytics import YOLO

# 从头创建一个新的 YOLO 模型。
# TODO: 这文件哪来的？下面不是加载模型了么？这文件干啥用的？创建的是什么？
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml 好像是这个？看不懂。。。
model = YOLO("yolo11n.yaml")

# 加载预训练的 YOLO 模型（推荐用于训练）
# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt 下载到项目根目录
model = YOLO("models/yolo11n.pt")

# 根据 coco8.yaml 数据集配置，训练模型 3 个周期
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
results = model.train(data="train_config/coco8.yaml", epochs=3)

# 评估模型在验证集上的性能
results = model.val()

# 使用模型对图像执行对象检测
# 这里的检测是识别图像中的物体，并给出物体的类别和位置。
results = model("https://ultralytics.com/images/bus.jpg")
# 若想要将识别结果的图片保存到本地，可以使用 results.show() 或 results.save()
results[0].show()

# 将模型导出为 ONNX 格式
success = model.export(format="onnx")


