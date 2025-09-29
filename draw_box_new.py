#coding:utf-8
import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# 全局变量路径配置
label_folder = 'E:/ProData/code/draw-YOLO-box/labels'
raw_images_folder = 'E:/ProData/code/draw-YOLO-box/raw_images'
save_images_folder = 'E:/ProData/code/draw-YOLO-box/save_image'
name_list_path = './name_list.txt'
classes_path = 'E:/ProData/code/draw-YOLO-box/classes-33.txt'


# 👇 修复版：确保加载到有效字体
def get_font(size):
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "Arial.ttf",
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size)
            print(f"✅ 使用字体: {path} | 大小: {size}")
            return font
        except Exception as e:
            continue
    print("⚠️  使用默认字体（可能模糊或错位）")
    return ImageFont.load_default()


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    tl = line_thickness or max(round(0.002 * (image.shape[0] + image.shape[1]) / 2), 1)
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        # 转 PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 字体
        font_size = max(int(tl * 10), 16)
        font = get_font(font_size)

        # 💥 修复错位核心：使用 anchor='lt' 让坐标表示左上角
        try:
            # 获取文字尺寸（相对于左上角 anchor）
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font, anchor='lt')
            tw = right - left
            th = bottom - top
        except TypeError:
            # 旧版 Pillow 不支持 anchor，用传统方式（会有轻微偏移）
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            tw = right - left
            th = bottom - top

        # 背景框左上角 = 检测框左上角
        bg_x1, bg_y1 = c1[0], c1[1] - th - 4
        bg_x2, bg_y2 = bg_x1 + tw + 4, c1[1]

        # 颜色转换 BGR → RGB
        if isinstance(color, (list, tuple)):
            color_rgb = tuple([int(c) for c in color][::-1])  # BGR to RGB
        else:
            color_rgb = (255, 255, 255)

        bg_color = tuple(max(0, c - 30) for c in color_rgb)
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)

        # 自动文字颜色
        brightness = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

        # 💥 修复错位：文字坐标 = 背景框左上角 + padding，使用 anchor='lt'
        text_x = bg_x1 + 2
        text_y = bg_y1 + 2

        try:
            # 描边
            #for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                #draw.text((text_x + dx, text_y + dy), label, fill=(0,0,0), font=font, anchor='lt')
            # 主文字
            draw.text((text_x, text_y), label, fill=text_color, font=font, anchor='lt')
        except TypeError:
            # 旧版 Pillow 不支持 anchor，使用默认（会有轻微偏移）
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        # 转回 OpenCV
        image[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return image


def draw_box_on_image(image_name, classes, colors, label_folder, raw_images_folder, save_images_folder):
    txt_path = os.path.join(label_folder, f'{image_name}.txt')
    print(f"处理: {image_name}")
    if image_name == '.DS_Store':
        return 0

    # 支持多格式图片
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_path = None
    for ext in extensions:
        candidate = os.path.join(raw_images_folder, image_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if not image_path:
        print(f"❌ 图片未找到: {image_name}")
        return 0

    # 保存为 PNG
    save_file_path = os.path.join(save_images_folder, f'{image_name}.png')

    if not os.path.exists(txt_path):
        print(f"⚠️ 标签文件不存在: {txt_path}")
        # 依然复制原图保存（可选）
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imwrite(save_file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return 0

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 图像读取失败: {image_path}")
        return 0

    try:
        height, width = image.shape[:2]
    except:
        print('❌ 无图像尺寸信息')
        return 0

    box_number = 0
    for line in lines:
        staff = line.strip().split()
        if len(staff) < 5:
            continue
        try:
            class_idx = int(staff[0])
            if class_idx >= len(classes):
                continue

            # YOLO 格式转像素坐标
            x_center = float(staff[1]) * width
            y_center = float(staff[2]) * height
            w = float(staff[3]) * width
            h = float(staff[4]) * height

            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)

            # 画框 + 标签
            plot_one_box([x1, y1, x2, y2], image, color=colors[class_idx], label=classes[class_idx], line_thickness=2)
            box_number += 1
        except Exception as e:
            print(f"❌ 处理行出错: {line.strip()} | 错误: {e}")
            continue

    # 保存图像（PNG 无损）
    cv2.imwrite(save_file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"✅ 保存: {save_file_path} | 框数量: {box_number}")
    return box_number


def make_name_list(raw_images_folder, name_list_path):
    image_files = os.listdir(raw_images_folder)
    valid_names = []
    for fname in image_files:
        if fname == '.DS_Store':
            continue
        name, ext = os.path.splitext(fname)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            valid_names.append(name)

    with open(name_list_path, 'w', encoding='utf-8') as f:
        for name in valid_names:
            f.write(name + '\n')
    print(f"📝 已生成 {len(valid_names)} 个文件名到 {name_list_path}")


if __name__ == '__main__':
    # 生成文件名列表
    make_name_list(raw_images_folder, name_list_path)

    # 读取类别名
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    # 固定颜色（确保每次运行一致）
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # 读取要处理的图片名
    with open(name_list_path, 'r', encoding='utf-8') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]

    # 开始处理
    box_total = 0
    image_total = 0
    for image_name in image_names:
        box_num = draw_box_on_image(image_name, classes, colors, label_folder, raw_images_folder, save_images_folder)
        box_total += box_num
        image_total += 1
        print(f'📊 总计: 框数={box_total}, 图片数={image_total}')