from ultralytics import YOLO
import cv2

# 1. 加载训练好的模型（替换为你的模型路径）
model = YOLO("runs/detect/train3/weights/best.pt")  # 例如: "runs/detect/train/weights/best.pt"

# 2. 读取待检测图片
image_path = "l.jpg"
image = cv2.imread(image_path)

# 3. 执行推理（自动处理多目标检测）
results = model(image)  # 返回包含所有检测结果的列表

# 4. 可视化结果（绘制边界框和标签）
for result in results:
    # 绘制到原图
    annotated_image = result.plot()

    # 保存或显示结果
    cv2.imwrite("output.jpg", annotated_image)
    cv2.imshow("Detection", annotated_image)
    cv2.waitKey(0)

print(f"检测到 {len(results[0].boxes)} 个物体")