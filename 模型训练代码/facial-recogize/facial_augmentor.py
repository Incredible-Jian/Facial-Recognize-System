import cv2
import numpy as np
import random
from PIL import Image
import os
os.environ['ALBUMENTATIONS_CHECK_VERSION'] = '0'
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FacialExpressionAugmentor:
    def __init__(self, image_size=640):
        """
        面部表情专用增强器

        参数:
            image_size: 输出图像尺寸
        """
        self.image_size = image_size

        # 基础增强
        self.base_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # 表情专用增强
        self.expression_augmentations = [
            self.add_eyeglasses,
            self.add_facial_occlusion,
            self.apply_expression_distortion,
            self.adjust_facial_lighting
        ]

    def __call__(self, image, bboxes=None, labels=None):
        """
        应用增强

        参数:
            image: 输入图像 (PIL Image 或 numpy数组)
            bboxes: 边界框 [x_min, y_min, x_max, y_max]
            labels: 类别标签

        返回:
            增强后的图像和标注
        """
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 应用表情专用增强
        if random.random() > 0.3:  # 70%概率应用专用增强
            aug_func = random.choice(self.expression_augmentations)
            image = aug_func(image)

        # 应用基础变换
        transformed = self.base_transform(image=image, bboxes=bboxes or [], class_labels=labels or [])

        # 返回结果
        return {
            'image': transformed['image'],
            'bboxes': transformed['bboxes'],
            'labels': transformed['class_labels']
        }

    def add_eyeglasses(self, image):
        """添加眼镜遮挡"""
        # 人脸检测简化版 - 假设人脸在图像中心
        h, w = image.shape[:2]
        face_center = (w//2, h//3)

        # 创建眼镜形状
        glass_width = int(w * 0.4)
        glass_height = int(h * 0.1)

        # 眼镜位置
        left_glass = (face_center[0] - glass_width//2, face_center[1] - glass_height//2)
        right_glass = (face_center[0] + glass_width//4, face_center[1] - glass_height//2)

        # 绘制眼镜
        cv2.rectangle(image,
                      (left_glass[0], left_glass[1]),
                      (left_glass[0] + glass_width//3, left_glass[1] + glass_height),
                      (0, 0, 0), -1)

        cv2.rectangle(image,
                      (right_glass[0], right_glass[1]),
                      (right_glass[0] + glass_width//3, right_glass[1] + glass_height),
                      (0, 0, 0), -1)

        # 连接桥
        cv2.line(image,
                 (left_glass[0] + glass_width//3, left_glass[1] + glass_height//2),
                 (right_glass[0], left_glass[1] + glass_height//2),
                 (0, 0, 0), 3)

        return image

    def add_facial_occlusion(self, image):
        """添加随机面部遮挡"""
        h, w = image.shape[:2]

        # 随机遮挡类型
        occlusion_type = random.choice(['mask', 'hand', 'hair', 'object'])

        # 人脸位置
        face_x = w // 2
        face_y = h // 3
        face_w = int(w * 0.4)
        face_h = int(h * 0.5)

        if occlusion_type == 'mask':
            # 口罩
            mask_y = face_y + face_h * 0.6
            mask_h = face_h * 0.2
            cv2.rectangle(image,
                          (int(face_x - face_w*0.4), int(mask_y)),
                          (int(face_x + face_w*0.4), int(mask_y + mask_h)),
                          (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

        elif occlusion_type == 'hand':
            # 手部遮挡
            points = np.array([
                [face_x, face_y + face_h*0.3],
                [face_x - face_w*0.3, face_y + face_h*0.1],
                [face_x - face_w*0.4, face_y + face_h*0.4],
                [face_x - face_w*0.3, face_y + face_h*0.6],
                [face_x, face_y + face_h*0.5]
            ], np.int32)
            cv2.fillPoly(image, [points], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        elif occlusion_type == 'hair':
            # 头发遮挡
            points = np.array([
                [face_x - face_w*0.5, face_y - face_h*0.1],
                [face_x - face_w*0.3, face_y + face_h*0.3],
                [face_x, face_y + face_h*0.2],
                [face_x + face_w*0.3, face_y + face_h*0.3],
                [face_x + face_w*0.5, face_y - face_h*0.1]
            ], np.int32)
            cv2.fillPoly(image, [points], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        else:  # object
            # 随机物体遮挡
            obj_x = random.randint(int(face_x - face_w*0.3), int(face_x + face_w*0.3))
            obj_y = random.randint(int(face_y + face_h*0.2), int(face_y + face_h*0.6))
            obj_size = random.randint(10, min(w, h)//5)

            shape = random.choice(['circle', 'rectangle', 'triangle'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            if shape == 'circle':
                cv2.circle(image, (obj_x, obj_y), obj_size, color, -1)
            elif shape == 'rectangle':
                cv2.rectangle(image,
                              (obj_x - obj_size, obj_y - obj_size),
                              (obj_x + obj_size, obj_y + obj_size),
                              color, -1)
            else:  # triangle
                points = np.array([
                    [obj_x, obj_y - obj_size],
                    [obj_x - obj_size, obj_y + obj_size],
                    [obj_x + obj_size, obj_y + obj_size]
                ], np.int32)
                cv2.fillPoly(image, [points], color)

        return image

    def apply_expression_distortion(self, image):
        """应用表情扭曲增强"""
        h, w = image.shape[:2]

        # 创建变形网格
        grid_x = np.tile(np.linspace(0, w, 15), (15, 1)).astype(np.float32)
        grid_y = np.tile(np.linspace(0, h, 15).reshape(15, 1), (1, 15)).astype(np.float32)

        # 随机扭曲中心点 (人脸中心)
        center_x = w // 2
        center_y = h // 3

        # 扭曲强度
        strength = random.uniform(0.8, 1.2)

        # 应用径向扭曲
        for i in range(15):
            for j in range(15):
                dx = grid_x[i, j] - center_x
                dy = grid_y[i, j] - center_y

                distance = np.sqrt(dx*dx + dy*dy)
                if distance > 0:
                    factor = 1.0 / (1.0 + distance * 0.001 * strength)

                    grid_x[i, j] = center_x + dx * factor
                    grid_y[i, j] = center_y + dy * factor

        # 应用网格变形
        map_x = cv2.resize(grid_x, (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(grid_y, (w, h), interpolation=cv2.INTER_LINEAR)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    def adjust_facial_lighting(self, image):
        """调整面部光照条件"""
        # 转换为HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 随机光照调整
        lighting_type = random.choice(['side', 'top', 'bottom', 'spot'])

        # 创建光照蒙版
        mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
        h, w = mask.shape

        # 人脸位置
        face_center = (w//2, h//3)
        face_radius = min(w, h) // 4

        if lighting_type == 'side':
            # 侧光
            for y in range(h):
                for x in range(w):
                    mask[y, x] = min(1.0, max(0.5, x / w))
        elif lighting_type == 'top':
            # 顶光
            for y in range(h):
                for x in range(w):
                    mask[y, x] = min(1.0, max(0.5, 1.0 - y / h))
        elif lighting_type == 'bottom':
            # 底光
            for y in range(h):
                for x in range(w):
                    mask[y, x] = min(1.0, max(0.5, y / h))
        else:  # spot
            # 聚光灯
            for y in range(h):
                for x in range(w):
                    distance = np.sqrt((x - face_center[0])**2 + (y - face_center[1])**2)
                    mask[y, x] = max(0.3, 1.0 - distance / (face_radius * 2))

        # 应用光照调整
        intensity = random.uniform(0.7, 1.5)  # 光照强度
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * mask * intensity, 0, 255).astype(np.uint8)

        # 转换回BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化增强器
    augmentor = FacialExpressionAugmentor(image_size=640)

    # 加载图像
    img_path = "./dataset/train/images/0a9d34ff118aca8feadb6a3f3b07e7d6f5f474b5f88c217c6d89895682b10472_-_20_-_-_20201208104133-005-010_jpg.rf.a8e209fe3fa0db1208dac2c3a76270bb.jpg"
    image = Image.open(img_path)

    # 应用增强
    augmented = augmentor(image)

    # 可视化结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # 转换张量回图像
    aug_img = augmented['image'].permute(1, 2, 0).numpy()
    aug_img = (aug_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
    plt.imshow(aug_img)
    plt.title("Augmented Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()