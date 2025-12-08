import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from tqdm import tqdm
from model import vgg

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class YOLOClassificationDataset:
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        """增强鲁棒性的数据集类"""
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图片目录不存在: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"标签目录不存在: {label_dir}")

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.img_files = []
        self.labels = []

        try:
            for f in os.listdir(img_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    label_file = os.path.splitext(f)[0] + '.txt'
                    label_path = os.path.join(label_dir, label_file)
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as lf:
                            lines = lf.readlines()
                            if lines:
                                try:
                                    class_id = int(lines[0].strip().split()[0])
                                    self.img_files.append(f)
                                    self.labels.append(class_id)
                                except (IndexError, ValueError) as e:
                                    print(f"警告：跳过文件 {label_path}，解析错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载数据集时出错: {e}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            # 返回一个空图像或跳过
            fake_image = torch.zeros(3, 224, 224)  # 假设输入大小是224x224
            return fake_image, 0  # 返回默认类别


def load_model(model_path, num_classes, device):
    """加载VGG模型并处理可能的键不匹配问题"""
    model = vgg(model_name="vgg16", num_classes=num_classes, init_weights=False).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # 处理不同保存方式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 处理多GPU训练保存的模型
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def evaluate_model(model, dataloader, device, class_names):
    """评估模型并返回预测结果"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), class_names


def plot_confusion_matrix(labels, preds, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_metrics(labels, preds, class_names):
    """绘制各类别评估指标"""
    cm = confusion_matrix(labels, preds)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # 避免除以0

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='salmon')
    plt.bar(x + width, f1, width, label='F1-Score', color='lightgreen')

    plt.title('Classification Metrics per Class', fontsize=16)
    plt.xlabel('Emotion Classes', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')

    for i in range(len(class_names)):
        plt.text(i - width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center')
        plt.text(i, recall[i] + 0.02, f"{recall[i]:.2f}", ha='center')
        plt.text(i + width, f1[i] + 0.02, f"{f1[i]:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig('class_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(labels, probs, class_names):
    """绘制多类别ROC曲线"""
    y_true = label_binarize(labels, classes=np.arange(len(class_names)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                    'purple', 'pink', 'brown', 'gray'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (AUC = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multi-Class ROC Curve', fontsize=16)
    plt.legend(loc="lower right", prop={'size': 10})
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(model, dataloader, device, class_names, num_samples=8):
    """可视化样本预测结果"""
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(15, 15))

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                if images_so_far >= num_samples:
                    break

                ax = plt.subplot((num_samples + 1) // 2, 2, images_so_far + 1)
                ax.axis('off')

                # 反归一化图像
                img = images.cpu().data[j].numpy().transpose((1, 2, 0))
                img = img * 0.5 + 0.5  # 根据train.py中的Normalize参数反归一化
                img = np.clip(img, 0, 1)

                true_label = class_names[labels[j]]
                pred_label = class_names[preds[j]]
                color = 'green' if true_label == pred_label else 'red'

                ax.imshow(img)
                ax.set_title(f'真实: {true_label}\n预测: {pred_label}', color=color, fontsize=12)
                images_so_far += 1

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_classification_report(labels, preds, class_names):
    """生成并保存分类报告"""
    report = classification_report(labels, preds, target_names=class_names, digits=3)
    print("Classification Report:")
    print(report)

    with open('classification_report.txt', 'w') as f:
        f.write(report)


def main():
    try:
        # 配置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(current_dir, "dataset")
        model_path = os.path.join(current_dir, "vgg16Net.pth")

        # 检查关键路径
        required_paths = {
            "dataset": data_root,
            "valid_images": os.path.join(data_root, "valid", "images"),
            "valid_labels": os.path.join(data_root, "valid", "labels"),
            "class_indices": os.path.join(current_dir, "class_indices.json"),
            "model": model_path
        }

        # 验证所有必要路径
        for name, path in required_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"所需路径不存在: {name} - {path}")

        print("所有必要路径验证通过，开始加载数据...")

        # 加载类别名称
        with open(required_paths["class_indices"], 'r') as f:
            class_indices = json.load(f)
        class_names = [class_indices[str(i)] for i in range(len(class_indices))]
        num_classes = len(class_names)

        # 设备设置
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 数据预处理 (与train.py中的验证集transform一致)
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 创建验证集
        val_dataset = YOLOClassificationDataset(
            required_paths["valid_images"],
            required_paths["valid_labels"],
            class_names,
            transform=data_transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        # 加载模型
        print("加载模型中...")
        model = load_model(model_path, num_classes, device)
        print("模型加载成功")

        # 评估模型
        print("开始评估模型...")
        labels, preds, probs, class_names = evaluate_model(model, val_loader, device, class_names)

        # 生成并保存结果
        generate_classification_report(labels, preds, class_names)
        plot_confusion_matrix(labels, preds, class_names)
        plot_class_metrics(labels, preds, class_names)
        plot_roc_curve(labels, probs, class_names)
        plot_sample_predictions(model, val_loader, device, class_names)

        print("评估完成，结果已保存到当前目录")

    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请检查以下内容：")
        print("1. 数据集路径是否正确")
        print("2. 必要文件是否存在（模型文件、类别索引文件）")
        print("3. 文件权限是否正常")
        print(f"当前工作目录：{os.getcwd()}")


if __name__ == '__main__':
    main()