import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

from model import vgg


class YOLOClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        """
        Args:
            img_dir: 图片目录 (e.g. "dataset/train/images")
            label_dir: 标签目录 (e.g. "dataset/train/labels")
            class_names: 类别名称列表
            transform: 数据增强
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        # 获取所有图片文件
        self.img_files = [f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]

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
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 配置参数
    data_root = "dataset"  # 修改为你的数据集根目录
    class_names = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Sleepy", "Surprised"]  # 你的类别名称

    # 创建数据集
    train_dataset = YOLOClassificationDataset(
        img_dir=os.path.join(data_root, "train", "images"),
        label_dir=os.path.join(data_root, "train", "labels"),
        class_names=class_names,
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    val_dataset = YOLOClassificationDataset(
        img_dir=os.path.join(data_root, "valid", "images"),
        label_dir=os.path.join(data_root, "valid", "labels"),
        class_names=class_names,
        transform=data_transform["val"]
    )
    val_num = len(val_dataset)

    # 保存类别映射
    cla_dict = train_dataset.class_to_idx
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=nw)

    validate_loader = DataLoader(val_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=9, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()