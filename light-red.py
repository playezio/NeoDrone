# light-red.py

import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import argparse
import re
from collections import defaultdict

def extract_timestamp_from_filename(filename: str):
    """
    从文件名提取时间戳（假设格式为: prefix_1652345678_123.jpg）
    返回浮点秒（如 1652345678.123）
    若无法提取，返回 None
    """
    stem = Path(filename).stem
    # 尝试匹配末尾的时间戳模式：数字_三位毫秒
    match = re.search(r'_(\d+)_(\d{3})$', stem)
    if match:
        sec = int(match.group(1))
        ms = int(match.group(2))
        return sec + ms / 1000.0
    return None

def time_sync_check(img_light_path, img_red_path, max_diff_ms=10):
    """
    验证两图像时间差是否 ≤ max_diff_ms 毫秒
    若无法提取时间戳，默认返回 True（依赖硬件同步）
    """
    t1 = extract_timestamp_from_filename(img_light_path.name)
    t2 = extract_timestamp_from_filename(img_red_path.name)
    if t1 is None or t2 is None:
        return True  # 信任 DJI 硬件同步
    return abs(t1 - t2) * 1000 <= max_diff_ms

def load_bboxes_from_xml(xml_path):
    """加载 Pascal VOC XML 中的边界框，返回 [(class_name, (cx, cy)), ...]"""
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        bboxes.append((name, (cx, cy)))
    return bboxes

def spatial_consistency_check(img_light, img_red, xml_light, xml_red, min_matches=150, max_center_dist=5):
    """
    执行空间一致性验证：
    1. SIFT 特征匹配 ≥ min_matches
    2. 同类目标中心点距离 ≤ max_center_dist
    """
    # Step 1: SIFT 匹配
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_light, None)
        kp2, des2 = sift.detectAndCompute(img_red, None)

        if des1 is None or des2 is None:
            return False

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < min_matches:
            return False
    except Exception:
        return False

    # Step 2: 边界框中心一致性（仅当两个XML都存在）
    bboxes_l = load_bboxes_from_xml(xml_light)
    bboxes_r = load_bboxes_from_xml(xml_red)

    if not bboxes_l and not bboxes_r:
        return True  # 无目标，视为一致
    if not bboxes_l or not bboxes_r:
        return False  # 一个有目标一个无，不一致

    # 构建类别到中心点的映射（简化：仅检查同类目标最近匹配）
    centers_l = defaultdict(list)
    centers_r = defaultdict(list)
    for name, center in bboxes_l:
        centers_l[name].append(center)
    for name, center in bboxes_r:
        centers_r[name].append(center)

    for cls in set(centers_l.keys()) | set(centers_r.keys()):
        list_l = centers_l.get(cls, [])
        list_r = centers_r.get(cls, [])
        if not list_l or not list_r:
            continue  # 某一类只在一侧出现，暂不视为错误（可能遮挡）
        # 简单最近邻检查
        for c_l in list_l:
            min_dist = min(np.linalg.norm(np.array(c_l) - np.array(c_r)) for c_r in list_r)
            if min_dist > max_center_dist:
                return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate & enforce consistency between visible and infrared image pairs in NEODrone.")
    parser.add_argument("--light_dir", type=str, required=True, help="Directory of visible light images (e.g., Images/light)")
    parser.add_argument("--red_dir", type=str, required=True, help="Directory of infrared images (e.g., Images/red)")
    parser.add_argument("--ann_dir", type=str, required=True, help="Directory of Pascal VOC XML annotations")
    parser.add_argument("--output_report", type=str, default="light_red_consistency_report.txt", help="Output report file")
    parser.add_argument("--keep_consistent_only", action="store_true", help="If set, move inconsistent pairs to 'rejected/' subdirs")

    args = parser.parse_args()

    light_dir = Path(args.light_dir)
    red_dir = Path(args.red_dir)
    ann_dir = Path(args.ann_dir)

    # 获取所有可见光图像（假设红外图像同名）
    light_imgs = {p.stem: p for p in light_dir.rglob("*.jpg")}
    red_imgs = {p.stem: p for p in red_dir.rglob("*.jpg")}

    common_stems = set(light_imgs.keys()) & set(red_imgs.keys())
    print(f"Found {len(common_stems)} potential image pairs.")

    if args.keep_consistent_only:
        (light_dir / "rejected").mkdir(exist_ok=True)
        (red_dir / "rejected").mkdir(exist_ok=True)

    passed = 0
    failed = 0
    report_lines = []

    for stem in sorted(common_stems):
        img_l = light_imgs[stem]
        img_r = red_imgs[stem]
        xml_l = ann_dir / (stem + ".xml")
        xml_r = ann_dir / (stem + ".xml")  # 同名标注

        # 时间同步检查
        if not time_sync_check(img_l, img_r):
            reason = "Time sync failed (>10ms)"
            failed += 1
            report_lines.append(f"{stem}: {reason}")
            if args.keep_consistent_only:
                img_l.rename(light_dir / "rejected" / img_l.name)
                img_r.rename(red_dir / "rejected" / img_r.name)
            continue

        # 加载图像
        im_l = cv2.imread(str(img_l), cv2.IMREAD_GRAYSCALE)
        im_r = cv2.imread(str(img_r), cv2.IMREAD_GRAYSCALE)
        if im_l is None or im_r is None:
            reason = "Image load failed"
            failed += 1
            report_lines.append(f"{stem}: {reason}")
            if args.keep_consistent_only:
                img_l.rename(light_dir / "rejected" / img_l.name)
                img_r.rename(red_dir / "rejected" / img_r.name)
            continue

        # 空间一致性检查
        if not spatial_consistency_check(im_l, im_r, xml_l, xml_r):
            reason = "Spatial consistency failed (SIFT<150 or bbox center dist>5px)"
            failed += 1
            report_lines.append(f"{stem}: {reason}")
            if args.keep_consistent_only:
                img_l.rename(light_dir / "rejected" / img_l.name)
                img_r.rename(red_dir / "rejected" / img_r.name)
            continue

        passed += 1

    # 输出报告
    with open(args.output_report, 'w') as f:
        f.write(f"NEODrone Light-Red Consistency Validation Report\n")
        f.write(f"Total pairs: {len(common_stems)}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n\n")
        if report_lines:
            f.write("Failed pairs:\n")
            f.write("\n".join(report_lines))
    
    print(f"Validation complete. Report saved to {args.output_report}")
    print(f"Passed: {passed}, Failed: {failed}")

    if args.keep_consistent_only:
        print("Inconsistent pairs moved to 'rejected/' subdirectories.")

if __name__ == "__main__":
    main()