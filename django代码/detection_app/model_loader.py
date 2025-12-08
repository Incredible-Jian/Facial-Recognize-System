from ultralytics import YOLO
from django.conf import settings

# 加载YOLO模型
model = YOLO(settings.YOLO_MODEL_PATH)