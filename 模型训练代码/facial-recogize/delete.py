import os
import random


def delete_random_files(train_dir, target_count):
    """
    删除 train 文件夹中的部分图片和对应的标签文件，直到文件数量达到目标数量。

    :param train_dir: 包含 images 和 labels 子文件夹的训练数据目录
    :param target_count: 目标文件数量
    """
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # 计算需要删除的文件数量
    current_count = len(image_files)
    if current_count <= target_count:
        print("当前文件数量已经小于或等于目标数量，无需删除。")
        return

    delete_count = current_count - target_count
    print(f"当前文件数量: {current_count}, 需要删除的文件数量: {delete_count}")

    # 随机选择要删除的文件
    delete_files = random.sample(image_files, delete_count)

    # 删除选中的文件及其对应的标签文件
    for file_name in delete_files:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + '.txt')

        # 删除图片文件
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"删除图片文件: {image_path}")

        # 删除对应的标签文件
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"删除标签文件: {label_path}")

    print(f"删除完成，剩余文件数量: {len(os.listdir(images_dir))}")


# 使用示例
train_directory = 'dataset/train'
target_file_count = 100  # 目标文件数量
delete_random_files(train_directory, target_file_count)