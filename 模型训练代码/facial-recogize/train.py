from ultralytics import YOLO
import torch
import os

# 设置环境变量
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

if __name__ == "__main__":
    # 加载模型
    model = YOLO("yolov8n.pt")

    # 训练配置（使用当前版本支持的参数）
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=320,
        batch=8,
        workers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu',

        # 优化器参数
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # 数据增强（调整后的表情识别优化参数）
        hsv_h=0.01,  # 减少色调变化
        hsv_s=0.5,  # 降低饱和度变化
        hsv_v=0.4,
        degrees=10,  # 增加旋转范围
        translate=0.1,
        scale=0.2,
        shear=2.0,
        perspective=0.001,
        flipud=0.1,  # 减少上下翻转
        fliplr=0.5,  # 保持水平翻转

        # 高级增强
        mosaic=0.8,  # 提高mosaic概率
        mixup=0.1,
        copy_paste=0.1,

        # 损失函数权重（新版参数名）
        cls=1.5,  # 分类损失权重
        box=7.5,  # 边界框损失权重
        dfl=1.5,  # 分布焦点损失权重（替代obj）

        # 训练控制
        patience=20,
        cos_lr=True,
        close_mosaic=5,
        amp=True  # 自动混合精度
    )