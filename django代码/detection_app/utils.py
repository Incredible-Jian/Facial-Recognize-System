import os
import cv2
import time
import numpy as np
from django.conf import settings

def generate_unique_filename(filename):
    """生成唯一的文件名"""
    base, ext = os.path.splitext(filename)
    timestamp = int(time.time())
    return f"{base}_{timestamp}{ext}"

def process_frame(frame, model):
    """处理单个视频帧"""
    # 调整尺寸以提高性能
    frame = cv2.resize(frame, (640, 480))

    # 使用YOLO模型进行识别
    results = model(frame)

    # 绘制结果
    return results[0].plot()

def get_emotion_stats(results, model):
    """从结果中提取表情统计"""
    emotion_counts = {}
    for box in results[0].boxes:
        cls = int(box.cls)
        emotion = model.names[cls]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    return emotion_counts