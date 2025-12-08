from ultralytics import YOLO
import numpy as np


def calculate_class_accuracy(confusion_matrix, class_names):
    """计算并打印每个类别的准确率"""
    # 计算每个类别的正确预测数和总样本数
    correct = np.diag(confusion_matrix)
    total = confusion_matrix.sum(axis=1)

    print("\n各个表情类别的准确率:")
    print(f"{'表情类别':<15}{'准确率':<10}{'样本数'}")
    print("-" * 35)

    for i, name in enumerate(class_names):
        acc = correct[i] / total[i] if total[i] > 0 else 0
        print(f"{name:<15}{acc:.4f}{'':<6}{int(total[i])}")


if __name__ == "__main__":
    # 加载训练好的模型
    model = YOLO("runs/detect/train4/weights/best.pt")

    # 在验证集上评估
    results = model.val(split='val')

    # 获取混淆矩阵和类别名称
    confusion_matrix = results.confusion_matrix.matrix
    class_names = model.names

    # 计算并显示准确率
    calculate_class_accuracy(confusion_matrix, class_names)

    # 打印整体指标
    print("\n整体验证指标:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"分类准确率: {results.cls.accuracy:.4f}")