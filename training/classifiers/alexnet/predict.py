"""AlexNet 单张图片表情分类预测。

用法（在仓库根目录执行）：
    python training/classifiers/alexnet/predict.py
    python training/classifiers/alexnet/predict.py --image path/to/face.jpg
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DATA_ROOT = SCRIPT_DIR.parents[1] / "dataset"


def default_weights():
    shipped = ROOT / "weights" / "AlexNet_9classes.pth"
    trained = SCRIPT_DIR / "AlexNet_9classes.pth"
    return shipped if shipped.exists() else trained


def parse_args():
    parser = argparse.ArgumentParser(description="AlexNet 表情分类预测")
    parser.add_argument("--image", default=None, help="待预测图片；缺省时使用训练集第一张")
    parser.add_argument("--weights", default=str(default_weights()), help="模型权重路径")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_path = args.image
    if image_path is None:
        candidates = sorted((DATA_ROOT / "train" / "images").glob("*"))
        if not candidates:
            raise FileNotFoundError("训练集目录为空，请通过 --image 指定图片")
        image_path = candidates[0]

    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)

    img_tensor = data_transform(img).unsqueeze(dim=0)

    with open(SCRIPT_DIR / "class_indices.json", "r") as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=len(class_indict)).to(device)
    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"权重文件不存在：{args.weights}\n"
            "请先运行 python training/classifiers/alexnet/train.py 训练模型，"
            "或从 GitHub Releases 下载 AlexNet_9classes.pth 放到 weights/ 目录。"
        )
    model.load_state_dict(torch.load(args.weights, map_location=device))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy()
    )
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(
            class_indict[str(i)], predict[i].numpy()
        ))
    plt.show()


if __name__ == "__main__":
    main()
