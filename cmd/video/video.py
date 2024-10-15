import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('models/yolo11n.pt')

# 打开摄像头
# 0表示默认摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行YOLO11推理
        results = model(frame)

        # 在帧上绘制注释
        annotated_frame = results[0].plot()

        # 利用 cv2 显示带注释的帧。
        cv2.imshow("YOLOv11 Inference", annotated_frame)

        # 按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果没有成功读取帧，退出循环
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()