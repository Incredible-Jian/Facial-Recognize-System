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
from model import resnet34

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class YOLOClassificationDataset:
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.img_files = []
        self.labels = []

        # 收集所有有效样本和标签
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                label_file = os.path.splitext(f)[0] + '.txt'
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    with open(label_path, 'r') as lf:
                        lines = lf.readlines()
                        if lines:
                            class_id = int(lines[0].strip().split()[0])
                            self.img_files.append(f)
                            self.labels.append(class_id)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(model_path, num_classes, device):
    """加载模型并处理可能的键不匹配问题"""
    model = resnet34(num_classes=num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # 处理不同保存方式
    if 'model_state_dict' in checkpoint:
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
    cm = confusion_matrix(labels, preds)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)

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
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
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
    report = classification_report(labels, preds, target_names=class_names, digits=3)
    print("Classification Report:")
    print(report)

    # 保存报告到文件
    with open('classification_report.txt', 'w') as f:
        f.write(report)


def main():
    # 配置参数
    data_root = "dataset"
    model_path = "resnet34_yolo.pth"

    # 加载类别名称
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]
    num_classes = len(class_names)

    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建验证集
    val_dataset = YOLOClassificationDataset(
        os.path.join(data_root, "valid", "images"),
        os.path.join(data_root, "valid", "labels"),
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
    model = load_model(model_path, num_classes, device)

    # 评估模型
    labels, preds, probs, class_names = evaluate_model(model, val_loader, device, class_names)

    # 生成并保存结果
    generate_classification_report(labels, preds, class_names)
    plot_confusion_matrix(labels, preds, class_names)
    plot_class_metrics(labels, preds, class_names)
    plot_roc_curve(labels, probs, class_names)
    plot_sample_predictions(model, val_loader, device, class_names)


if __name__ == '__main__':
    main()