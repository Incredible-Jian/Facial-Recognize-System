from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import random

def generate_advanced_logo(output_path="logo.png", size=(800, 800)):
    # 创建渐变背景 - 蓝紫色调
    base = Image.new('RGB', size, (25, 40, 100))
    draw = ImageDraw.Draw(base)

    # 添加光晕效果
    for i in range(5):
        x, y = random.randint(0, size[0]), random.randint(0, size[1])
        r = random.randint(100, 300)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(100, 80, 180, 50))

    # 模糊背景
    base = base.filter(ImageFilter.GaussianBlur(20))

    # 添加中心元素
    center_img = Image.new('RGBA', (400, 400))
    center_draw = ImageDraw.Draw(center_img)

    # 创建表情环
    emotions = ["😊", "😢", "😠", "😲", "😞", "😍"]
    for i, emoji in enumerate(emotions):
        try:
            emoji_font = ImageFont.truetype("seguiemj.ttf", 60)
        except:
            emoji_font = ImageFont.load_default(60)

        angle = i * (360/len(emotions))
        x = 200 + 150 * np.cos(np.radians(angle))
        y = 200 + 150 * np.sin(np.radians(angle))
        center_draw.text((x, y), emoji, font=emoji_font, anchor="mm")

    # 添加中心AI图标
    try:
        icon_font = ImageFont.truetype("arial.ttf", 120)
        center_draw.text((200, 200), "AI", fill="white", font=icon_font, anchor="mm")
    except:
        pass

    # 合并到背景
    base.paste(center_img, (size[0]//2-200, size[1]//2-200), center_img)

    # 添加文字
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 60)
        subtitle_font = ImageFont.truetype("arial.ttf", 30)
    except:
        title_font = ImageFont.load_default(60)
        subtitle_font = ImageFont.load_default(30)

    draw.text((size[0]//2, size[1]-180), "EmotionAI",
              fill="white", font=title_font, anchor="mm", stroke_width=2, stroke_fill=(0,0,50))

    draw.text((size[0]//2, size[1]-120), "Facial Expression Recognition System",
              fill=(200, 200, 255), font=subtitle_font, anchor="mm")

    # 添加光泽效果
    gloss = Image.new('RGBA', size, (0,0,0,0))
    gloss_draw = ImageDraw.Draw(gloss)
    gloss_draw.rectangle((0, 0, size[0], size[1]//3), fill=(255,255,255,30))
    base = Image.alpha_composite(base.convert('RGBA'), gloss)

    # 保存为PNG
    base.save(output_path, quality=95)
    print(f"高级Logo已生成: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_advanced_logo()