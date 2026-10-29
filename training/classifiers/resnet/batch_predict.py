"""ResNet34 批量图片表情分类预测。

用法（在仓库根目录执行）：
    python training/classifiers/resnet/batch_predict.py
    python training/classifiers/resnet/batch_predict.py --images path/to/folder
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DATA_ROOT = SCRIPT_DIR.parents[1] / "dataset"


def default_weights():
    shipped = ROOT / "weights" / "resNet34.pth"
    trained = SCRIPT_DIR / "resnet34_yolo.pth"
    return shipped if shipped.exists() else trained


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet34 批量表情分类预测")
    parser.add_argument(
        "--images",
        default=str(DATA_ROOT / "valid" / "images"),
        help="待预测图片文件夹",
    )
    parser.add_argument("--weights", default=str(default_weights()), help="模型权重路径")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小")
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

    imgs_root = Path(args.images)
    if not imgs_root.exists():
        raise FileNotFoundError(f"图片目录不存在：{imgs_root}")

    img_path_list = sorted(
        p for p in imgs_root.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    with open(SCRIPT_DIR / "class_indices.json", "r") as f:
        class_indict = json.load(f)

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"权重文件不存在：{args.weights}\n"
            "请先训练模型，或从 GitHub Releases 下载 resNet34.pth 放到 weights/ 目录。"
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

    batch_size = args.batch_size
    with torch.no_grad():
        for start in range(0, len(img_path_list), batch_size):
            batch_paths = img_path_list[start:start + batch_size]
            img_list = [data_transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch_img = torch.stack(img_list, dim=0)

            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for img_path, pro, cla in zip(batch_paths, probs, classes):
                print("image: {}  class: {}  prob: {:.3}".format(
                    img_path, class_indict[str(cla.item())], pro.item()
                ))


if __name__ == "__main__":
    main()
