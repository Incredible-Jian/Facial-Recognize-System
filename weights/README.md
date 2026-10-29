# 模型权重

| 文件 | 大小 | 用途 | 是否随仓库提供 |
| --- | ---: | --- | --- |
| `best.pt` | 6 MB | YOLOv8 九类表情检测，Web 端推理使用 | 是 |
| `resNet34.pth` | 82 MB | ResNet34 旧版 **5 分类**权重，不兼容九类映射 | 否，见 Releases |
| `AlexNet_9classes.pth` | 56 MB | AlexNet 九类表情分类 | 否，见 Releases |
| `vgg16Net.pth` | — | VGG16 表情分类 | 否，运行训练脚本生成 |

> `resNet34.pth` 的 `fc` 层输出为 5 类，与当前九类 `class_indices.json` 不匹配。预测/评估脚本会给出明确报错；如需九类 ResNet34，请运行 `training/classifiers/resnet/train.py` 重新训练。

## 下载分类模型权重

前往 [GitHub Releases](https://github.com/Incredible-Jian/Facial-Recognize-System/releases) 下载 `resNet34.pth` 与 `AlexNet_9classes.pth`，放到本目录即可。

## 重新训练

```bash
cd Facial-Recognize-System
pip install -r training/requirements.txt

python training/classifiers/resnet/train.py
python training/classifiers/alexnet/train.py
python training/classifiers/vgg/train.py
```

训练完成后，权重会保存在对应模型目录（`training/classifiers/<model>/`）。可以把它们移动或复制到 `weights/` 并改名，预测脚本会自动优先读取 `weights/` 下的同名文件。

## 关于模型规模

`best.pt` 的骨干网络为 YOLOv8n 规模（`depth_multiple=0.33`、`width_multiple=0.25`，约 3M 参数）。项目文档中的“YOLOv8m”是训练方案名称；如需更大模型，把 `training/yolo/train.py` 中的 `YOLO("yolov8n.pt")` 改为 `YOLO("yolov8m.pt")` 后重新训练。
