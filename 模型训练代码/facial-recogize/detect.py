from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

class EmotionDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = [
            'Angry', 'Contempt', 'Disgust', 'Fear',
            'Happy', 'Neutral', 'Sad', 'Sleepy', 'Surprised'
        ]
        # 表情颜色映射
        self.color_map = {
            'Angry': (0, 0, 255),       # 红色
            'Contempt': (255, 0, 255),   # 紫色
            'Disgust': (0, 128, 0),      # 深绿色
            'Fear': (255, 165, 0),       # 橙色
            'Happy': (0, 255, 0),        # 绿色
            'Neutral': (128, 128, 128),  # 灰色
            'Sad': (0, 0, 128),          # 深蓝色
            'Sleepy': (128, 0, 128),     # 深紫色
            'Surprised': (255, 255, 0)   # 黄色
        }
        # 表情跟踪缓存
        self.track_history = defaultdict(lambda: [])

    def detect(self, frame):
        # 使用更高的置信度和IoU阈值
        results = self.model.track(
            frame,
            persist=True,  # 启用目标跟踪
            verbose=False,
            conf=0.5,      # 置信度阈值
            iou=0.6,       # IoU阈值
            imgsz=640
        )

        # 解析结果
        output = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls_idx, conf in zip(boxes, track_ids, classes, confs):
                x, y, w, h = box
                emotion = self.class_names[int(cls_idx)]

                # 更新跟踪历史
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:  # 保留最近30个点
                    track.pop(0)

                output.append({
                    'track_id': track_id,
                    'bbox': [x, y, w, h],
                    'emotion': emotion,
                    'confidence': float(conf)
                })

        return output, results[0].plot()

    def draw_results(self, frame, output):
        # 绘制检测结果
        for obj in output:
            x, y, w, h = obj['bbox']
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)

            # 绘制边界框
            color = self.color_map.get(obj['emotion'], (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制表情标签
            label = f"{obj['track_id']}: {obj['emotion']} {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 绘制跟踪轨迹
            track = self.track_history[obj['track_id']]
            points = np.array(track, dtype=np.int32)
            cv2.polylines(frame, [points], False, color, 2)

        return frame

if __name__ == "__main__":
    # 初始化检测器
    detector = EmotionDetector("best.pt")

    # 检测视频
    file_path = "./dataset/valid/images"
    cap = cv2.VideoCapture(file_path if file_path.endswith(('.mp4', '.avi')) else 0)

    # 创建输出视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检测表情
        results, annotated_frame = detector.detect(frame)

        # 保存和显示
        out.write(annotated_frame)
        cv2.imshow('Emotion Detection', annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC退出
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()