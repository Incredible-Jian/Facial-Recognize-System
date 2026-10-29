"""不启动 Django 的最简摄像头表情检测演示。

用法（在仓库根目录执行）：
    python scripts/webcam_demo.py
按 q 退出。
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "best.pt"


def main():
    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备是否被占用")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        cv2.imshow("yolov8 emotion", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
