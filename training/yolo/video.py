"""摄像头实时表情检测（按 q 退出）。

用法（在仓库根目录执行）：
    python training/yolo/video.py
    python training/yolo/video.py --source 0
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 实时表情检测")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--source", default="0", help="摄像头编号或视频文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源：{args.source}")

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
