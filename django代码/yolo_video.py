
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    # ret 是否成功读取，frame读取到图像
    ret,frame = cap.read()
    if ret:
        res = model(frame) # res是一个列表
        ann = res[0].plot()
        # 将结果绘制在图像上，生成一共包含检测框和标签的图像
        cv2.imshow('yolov8',ann)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
cv2.destroyAllWindows()
cv2.release()
