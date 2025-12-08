import os
import torch
import torch.nn as nn
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建9分类模型
    net = resnet34(num_classes=9).to(device)

    # 加载预训练权重（部分加载）
    model_weight_path = "resnet34-pre.pth"
    if os.path.exists(model_weight_path):
        pretrained_dict = torch.load(model_weight_path, map_location=device)
        model_dict = net.state_dict()

        # 过滤不匹配的键
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        # 更新模型字典
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict, strict=False)
        print("部分权重加载成功（忽略不匹配的层）")
    else:
        print(f"警告: 预训练权重文件不存在 {model_weight_path}")

    print(net.fc)


if __name__ == '__main__':
    main()