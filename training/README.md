# 模型训练

本目录包含 YOLOv8 目标检测模型与 ResNet34 / AlexNet / VGG16 分类模型的训练、预测、评估代码。

## 环境

```bash
pip install -r training/requirements.txt
```

有 NVIDIA GPU 时建议安装与 CUDA 匹配的 PyTorch 版本，否则默认使用 CPU（训练会较慢）。

## 目录

```text
training/
├── dataset/                  # YOLO 格式数据集
│   ├── data.yaml             # 相对路径配置，可直接使用
│   ├── train/{images,labels}
│   ├── valid/{images,labels}
│   └── test/{images,labels}
├── yolo/                     # YOLOv8 检测
│   ├── train.py
│   ├── eval.py
│   ├── predict.py
│   ├── video.py
│   └── detect.py
├── classifiers/
│   ├── resnet/               # ResNet34
│   ├── alexnet/              # AlexNet
│   └── vgg/                  # VGG16
├── augmentation/             # 表情专用数据增强
└── tools/                    # 数据集裁剪、显存检查、标签可视化
```

## YOLOv8 检测

```bash
# 训练（默认 yolov8n.pt，可改为 yolov8m.pt）
python training/yolo/train.py

# 评估
python training/yolo/eval.py

# 单张图片预测
python training/yolo/predict.py --image path/to/face.jpg

# 摄像头实时检测
python training/yolo/video.py
```

训练输出位于 `training/runs/`，其中 `weights/best.pt` 可复制到仓库根目录的 `weights/` 供 Web 端使用。

## 分类模型

```bash
python training/classifiers/resnet/train.py
python training/classifiers/resnet/predict.py
python training/classifiers/resnet/eval.py

python training/classifiers/alexnet/train.py
python training/classifiers/alexnet/predict.py
python training/classifiers/alexnet/eval.py

python training/classifiers/vgg/train.py
python training/classifiers/vgg/predict.py
python training/classifiers/vgg/eval.py
```

- 训练脚本自动读取 `training/dataset/`，无需手动改路径。
- 预测脚本默认优先读取 `weights/` 下的权重，其次读取模型目录下训练产生的权重；也可用 `--weights` 指定。
- 评估脚本会在对应模型目录生成混淆矩阵、ROC 曲线、分类报告等文件。
- `class_indices.json` 统一为“索引 → 类别名”格式。

## 数据集

`dataset/data.yaml` 使用相对路径：

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 9
names: ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Sleepy', 'Surprised']
```

仓库内置每个划分 100 张的样例数据。替换为自己的数据时，保持相同目录结构并确保图片与同名 `.txt` 标签一一对应。

## 工具

```bash
# 随机裁剪训练集到指定数量（会删除文件，建议先备份）
python training/tools/delete.py --target 100 --dry-run
python training/tools/delete.py --target 100

# 查看 CUDA 显存占用
python training/tools/clear.py

# 按标签绘制检测框
python training/tools/image.py
```
