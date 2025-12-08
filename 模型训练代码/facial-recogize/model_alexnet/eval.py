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
from model import AlexNet
from tqdm import tqdm


class YOLOClassificationDataset:
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.valid_samples = []
        for img_file in self.img_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.valid_samples.append(img_file)

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_file = self.valid_samples[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    class_id = int(lines[0].strip().split()[0])
                    label = class_id

        if self.transform:
            image = self.transform(image)
        return image, label


def load_data_and_model(data_root="dataset", model_path="AlexNet_9classes.pth"):
    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别名称
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

    # 创建数据集
    val_dataset = YOLOClassificationDataset(
        img_dir=os.path.join(data_root, "valid", "images"),
        label_dir=os.path.join(data_root, "valid", "labels"),
        class_names=class_names,
        transform=data_transform
    )

    # 创建DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, val_loader, class_names, device


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
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


def plot_class_accuracy(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_acc, color='skyblue')
    plt.title('Classification Accuracy per Class', fontsize=16)
    plt.xlabel('Emotion Classes', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')

    # 在柱子上方添加准确率数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(labels, probs, class_names):
    # 二值化标签
    y_true = label_binarize(labels, classes=np.arange(len(class_names)))

    # 计算每个类的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制所有类的ROC曲线
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
    plt.title('ROC Curve for Multi-Class', fontsize=16)
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
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=12)
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
    # 加载数据和模型
    model, val_loader, class_names, device = load_data_and_model()

    # 评估模型
    labels, preds, probs, class_names = evaluate_model(model, val_loader, device, class_names)

    # 生成并打印分类报告
    generate_classification_report(labels, preds, class_names)

    # 可视化结果
    plot_confusion_matrix(labels, preds, class_names)
    plot_class_accuracy(labels, preds, class_names)
    plot_roc_curve(labels, probs, class_names)
    plot_sample_predictions(model, val_loader, device, class_names)


if __name__ == '__main__':
    main()