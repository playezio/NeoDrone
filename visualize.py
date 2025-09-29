# visualize.py

import os
import cv2
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import numpy as np

# 预定义30个类别的颜色（使用HSV生成可区分颜色，再转为BGR）
def generate_colors(num_classes):
    hsv_colors = [(i * 180 // num_classes, 255, 255) for i in range(num_classes)]
    bgr_colors = [tuple(map(int, cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0])) for hsv in hsv_colors]
    return bgr_colors

# NEODrone 的 30 个类别（顺序与数据集一致）
CLASS_NAMES = [
    "person", "tricycle", "car", "van", "box_truck", "truck", "bus", "liquefied_gas_truck",
    "eight_wheel_truck", "excavator", "crane", "bulldozer", "road_rescue_vehicle", "water_sprinkler",
    "power_tower", "ship", "armored_vehicle", "fire_truck", "fixed_wing_drone", "hexacopter",
    "quadcopter", "construction_site", "concrete_mixer", "electric_bicycle", "forklift", "pile_driver",
    "road_roller", "emergency_engineering_vehicle", "agricultural_machinery", "concrete_pumper"
]

COLORS = generate_colors(len(CLASS_NAMES))
CLASS_TO_COLOR = {name: COLORS[i] for i, name in enumerate(CLASS_NAMES)}

def parse_voc_xml(xml_path):
    """解析 Pascal VOC 格式的 XML 标注文件，返回目标列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        objects.append({
            'name': name,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    return objects

def draw_annotations(image, objects, show_class=True, show_bbox=True, font_scale=0.6, thickness=2):
    """在图像上绘制边界框和类别标签"""
    for obj in objects:
        name = obj['name']
        xmin, ymin, xmax, ymax = obj['bbox']
        color = CLASS_TO_COLOR.get(name, (0, 255, 0))  # 默认绿色

        if show_bbox:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        if show_class:
            label = name
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # 绘制背景矩形
            cv2.rectangle(image, (xmin, ymin - h - 5), (xmin + w, ymin), color, -1)
            cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return image

def visualize_single(image_path, xml_path, output_path=None, show=True, **draw_kwargs):
    """可视化单张图像及其标注"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return None

    if not xml_path.exists():
        print(f"Warning: Annotation file {xml_path} not found. Showing raw image.")
        objects = []
    else:
        objects = parse_voc_xml(xml_path)

    annotated_img = draw_annotations(img.copy(), objects, **draw_kwargs)

    if output_path:
        cv2.imwrite(str(output_path), annotated_img)
        print(f"Saved visualization to {output_path}")

    if show:
        cv2.imshow("NEODrone Visualization", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return annotated_img

def main():
    parser = argparse.ArgumentParser(description="Visualize NEODrone dataset images with Pascal VOC annotations.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing .jpg images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing .xml annotation files")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional: Directory to save visualized images")
    parser.add_argument("--image_name", type=str, default=None, help="Visualize a specific image (e.g., scene_001_00045.jpg)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize (if --image_name not set)")
    parser.add_argument("--no_show", action="store_true", help="Do not display images (useful for headless servers)")
    parser.add_argument("--no_class", action="store_true", help="Do not show class labels")
    parser.add_argument("--no_bbox", action="store_true", help="Do not draw bounding boxes")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    ann_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_name:
        img_path = image_dir / args.image_name
        xml_path = ann_dir / (img_path.stem + ".xml")
        visualize_single(
            img_path,
            xml_path,
            output_path=output_dir / (img_path.stem + "_vis.jpg") if output_dir else None,
            show=not args.no_show,
            show_class=not args.no_class,
            show_bbox=not args.no_bbox
        )
    else:
        # 随机选择 num_samples 张图像
        jpg_files = list(image_dir.rglob("*.jpg"))
        if not jpg_files:
            print(f"No .jpg files found in {image_dir}")
            return

        import random
        random.seed(42)
        selected = random.sample(jpg_files, min(args.num_samples, len(jpg_files)))

        for img_path in selected:
            xml_path = ann_dir / (img_path.stem + ".xml")
            out_path = output_dir / (img_path.name.replace(".jpg", "_vis.jpg")) if output_dir else None
            visualize_single(
                img_path,
                xml_path,
                output_path=out_path,
                show=not args.no_show,
                show_class=not args.no_class,
                show_bbox=not args.no_bbox
            )

if __name__ == "__main__":
    main()