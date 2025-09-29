# voc2coco.py

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse

# NEODrone 的 30 个类别（与数据集定义严格一致）
CLASS_NAMES = [
    "person", "tricycle", "car", "van", "box_truck", "truck", "bus", "liquefied_gas_truck",
    "eight_wheel_truck", "excavator", "crane", "bulldozer", "road_rescue_vehicle", "water_sprinkler",
    "power_tower", "ship", "armored_vehicle", "fire_truck", "fixed_wing_drone", "hexacopter",
    "quadcopter", "construction_site", "concrete_mixer", "electric_bicycle", "forklift", "pile_driver",
    "road_roller", "emergency_engineering_vehicle", "agricultural_machinery", "concrete_pumper"
]

def parse_voc_annotation(xml_path):
    """解析单个 VOC XML 文件，返回图像信息和目标列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_NAMES:
            continue  # 跳过未知类别（理论上不应出现）

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # COCO 使用 [x, y, width, height] 格式
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        # 面积
        area = bbox[2] * bbox[3]
        # 确保 bbox 为正
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue

        objects.append({
            'category_name': name,
            'bbox': bbox,
            'area': area
        })

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }

def voc_to_coco(annotation_dir, output_json, image_dir=None):
    """
    将整个 VOC 格式标注目录转换为 COCO JSON
    :param annotation_dir: 包含 .xml 文件的目录
    :param output_json: 输出的 COCO JSON 路径
    :param image_dir: 可选，用于验证图像是否存在（不影响转换）
    """
    ann_dir = Path(annotation_dir)
    xml_files = sorted(ann_dir.rglob("*.xml"))

    if not xml_files:
        raise ValueError(f"No .xml files found in {annotation_dir}")

    # 构建类别映射
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]
    category_name_to_id = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}

    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    for xml_file in tqdm(xml_files, desc="Converting VOC to COCO"):
        try:
            ann_info = parse_voc_annotation(xml_file)

            # 可选：检查图像是否存在
            if image_dir:
                img_path = Path(image_dir) / ann_info['filename']
                if not img_path.exists():
                    print(f"Warning: Image {img_path} not found. Skipping.")
                    continue

            image_info = {
                "id": image_id,
                "file_name": ann_info['filename'],
                "width": ann_info['width'],
                "height": ann_info['height']
            }
            images.append(image_info)

            for obj in ann_info['objects']:
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_name_to_id[obj['category_name']],
                    "bbox": obj['bbox'],
                    "area": obj['area'],
                    "iscrowd": 0,
                    "segmentation": []  # NEODrone 无分割标注
                }
                annotations.append(ann)
                annotation_id += 1

            image_id += 1

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)

    print(f"\nConversion completed!")
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"COCO JSON saved to: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Convert NEODrone VOC XML annotations to COCO JSON format.")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Path to directory containing .xml files")
    parser.add_argument("--output_json", type=str, default="instances.json", help="Output COCO JSON file path")
    parser.add_argument("--image_dir", type=str, default=None, help="Optional: Path to image directory for validation")

    args = parser.parse_args()

    voc_to_coco(
        annotation_dir=args.annotation_dir,
        output_json=args.output_json,
        image_dir=args.image_dir
    )

if __name__ == "__main__":
    main()