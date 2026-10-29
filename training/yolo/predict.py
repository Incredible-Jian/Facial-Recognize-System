"""单张图片表情检测。

用法（在仓库根目录执行）：
    python training/yolo/predict.py
    python training/yolo/predict.py --image path/to/face.jpg
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"
SAMPLE_DIR = ROOT / "training" / "dataset" / "test" / "images"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 单张图片表情检测")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--image", default=None, help="待检测图片；缺省时使用测试集第一张")
    parser.add_argument("--output", default="output.jpg", help="结果保存路径")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image
    if image_path is None:
        candidates = sorted(SAMPLE_DIR.glob("*"))
        if not candidates:
            raise FileNotFoundError(f"测试集目录为空：{SAMPLE_DIR}")
        image_path = str(candidates[0])

    model = YOLO(args.weights)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    results = model(image)
    annotated_image = results[0].plot()
    cv2.imwrite(args.output, annotated_image)

    print(f"图片: {image_path}")
    print(f"检测到 {len(results[0].boxes)} 个目标，结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
