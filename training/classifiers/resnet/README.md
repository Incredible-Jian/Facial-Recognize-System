# ResNet34 表情分类

## 文件

| 文件 | 说明 |
| --- | --- |
| `model.py` | ResNet34 网络结构 |
| `train.py` | 训练脚本（读取 YOLO 格式数据集） |
| `predict.py` | 单张图片预测 |
| `batch_predict.py` | 批量图片预测 |
| `eval.py` | 混淆矩阵、ROC、分类报告 |
| `load_weights.py` | 加载 ImageNet 预训练权重（可选） |
| `class_indices.json` | 索引到类别名的映射 |

## 使用

```bash
# 训练，权重输出到本目录 resnet34_yolo.pth
python training/classifiers/resnet/train.py

# 预测（默认优先读取 weights/resNet34.pth）
python training/classifiers/resnet/predict.py --image path/to/face.jpg

# 评估
python training/classifiers/resnet/eval.py
```

预训练权重 `resNet34.pth` 可从仓库 Releases 下载到 `weights/`。
