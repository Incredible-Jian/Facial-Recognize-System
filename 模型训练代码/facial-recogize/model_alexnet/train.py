import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import AlexNet  # 导入修改后的ResNet模型


class YOLOClassificationDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        """
        Args:
            img_dir: 图片目录 (e.g. "dataset/train/images")
            label_dir: 标签目录 (e.g. "dataset/train/labels")
            class_names: 类别名称列表 (e.g. ["cat", "dog"])
            transform: 数据增强
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        # 获取所有图片文件
        self.img_files = [f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 验证图片和标签匹配
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

        # 获取对应的标签文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        # 读取YOLO格式标签 (默认取第一个目标的类别)
        label = 0  # 默认类别
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    class_id = int(lines[0].strip().split()[0])
                    label = class_id

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 配置参数
    data_root = "dataset"
    class_names = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Sleepy", "Surprised"]  # 替换为你的实际类别
    num_classes = len(class_names)
    batch_size = 32
    epochs = 30
    lr = 0.001

    # 创建数据集
    train_dataset = YOLOClassificationDataset(
        img_dir=os.path.join(data_root, "train", "images"),
        label_dir=os.path.join(data_root, "train", "labels"),
        class_names=class_names,
        transform=data_transform["train"]
    )

    val_dataset = YOLOClassificationDataset(
        img_dir=os.path.join(data_root, "valid", "images"),
        label_dir=os.path.join(data_root, "valid", "labels"),
        class_names=class_names,
        transform=data_transform["val"]
    )

    # 保存类别映射
    with open('class_indices.json', 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # 初始化模型
    model = AlexNet(num_classes=num_classes)
    model.to(device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练循环
    best_acc = 0.0
    save_path = 'AlexNet_9classes.pth'

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{epochs}] Train')

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        scheduler.step()

        # 验证阶段
        model.eval()
        acc = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch [{epoch + 1}/{epochs}] Val')

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels).sum().item()

        val_accurate = acc / len(val_dataset)
        print(f'Epoch {epoch + 1}: Train Loss: {running_loss / len(train_loader):.3f}, '
              f'Val Acc: {val_accurate:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Training Finished')


if __name__ == '__main__':
    main()