"""ResNet34 单张图片表情分类预测。

用法（在仓库根目录执行）：
    python training/classifiers/resnet/predict.py
    python training/classifiers/resnet/predict.py --image path/to/face.jpg
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DATA_ROOT = SCRIPT_DIR.parents[1] / "dataset"


def default_weights():
    shipped = ROOT / "weights" / "resNet34.pth"
    trained = SCRIPT_DIR / "resnet34_yolo.pth"
    return shipped if shipped.exists() else trained


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet34 表情分类预测")
    parser.add_argument("--image", default=None, help="待预测图片；缺省时使用训练集第一张")
    parser.add_argument("--weights", default=str(default_weights()), help="模型权重路径")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_path = args.image
    if image_path is None:
        candidates = sorted((DATA_ROOT / "train" / "images").glob("*"))
        if not candidates:
            raise FileNotFoundError("训练集目录为空，请通过 --image 指定图片")
        image_path = candidates[0]

    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)

    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    json_path = SCRIPT_DIR / "class_indices.json"
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"权重文件不存在：{args.weights}\n"
            "请先运行 python training/classifiers/resnet/train.py 训练模型，"
            "或从 GitHub Releases 下载 resNet34.pth 放到 weights/ 目录。"
        )

    state_dict = torch.load(args.weights, map_location=device)
    checkpoint_classes = state_dict["fc.weight"].shape[0]
    if checkpoint_classes != len(class_indict):
        raise RuntimeError(
            f"权重文件是 {checkpoint_classes} 分类模型，"
            f"但 class_indices.json 包含 {len(class_indict)} 个类别。\n"
            "仓库中的 resNet34.pth 是旧版 5 分类权重，与九类表情映射不兼容；"
            "请运行 train.py 重新训练九类模型，或换用匹配的权重。"
        )

    model = resnet34(num_classes=checkpoint_classes).to(device)
    model.load_state_dict(state_dict)

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
