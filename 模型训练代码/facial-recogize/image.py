from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_boxes(img, boxes, class_names, class_colors, size=2):
    """优化绘制边界框函数"""
    img = np.array(img)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_id = int(box[4]) if len(box) > 4 else 0
        color = class_colors.get(class_names[cls_id], (123, 104, 238))

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, size)

        # 绘制类别标签
        label = class_names[cls_id]
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return Image.fromarray(img)

def parse_label(path, width, height):
    """优化标签解析函数"""
    boxes = []
    with open(path) as f:
        for line in f:
            data = line.strip().split()
            if len(data) < 5:
                continue

            cls_id = int(data[0])
            x_center = float(data[1]) * width
            y_center = float(data[2]) * height
            box_width = float(data[3]) * width
            box_height = float(data[4]) * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            boxes.append([x1, y1, x2, y2, cls_id])

    return boxes

if __name__ == '__main__':
    # 类别颜色映射
    class_colors = {
        'Angry': (0, 0, 255),
        'Contempt': (255, 0, 255),
        'Disgust': (0, 128, 0),
        'Fear': (255, 165, 0),
        'Happy': (0, 255, 0),
        'Neutral': (128, 128, 128),
        'Sad': (0, 0, 128),
        'Sleepy': (128, 0, 128),
        'Surprised': (255, 255, 0)
    }

    # 类别名称
    class_names = [
        'Angry', 'Contempt', 'Disgust', 'Fear',
        'Happy', 'Neutral', 'Sad', 'Sleepy', 'Surprised'
    ]

    # 加载图像
    img_path = './dataset/train/images/0a9d34ff118aca8feadb6a3f3b07e7d6f5f474b5f88c217c6d89895682b10472_-_20_-_-_20201208104106-002-011_jpg.rf.9ea3a3af10dc2f419dd5259380092136.jpg'
    img = Image.open(img_path)
    width, height = img.size

    # 解析标签
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    boxes = parse_label(label_path, width, height)

    # 绘制边界框
    result_img = draw_boxes(img, boxes, class_names, class_colors, size=3)

    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title('Facial Expression Detection')
    plt.show()