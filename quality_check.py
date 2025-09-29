# quality_check.py

import os
import cv2
import numpy as np
import argparse
import pandas as pd
from pathlib import Path

def calculate_laplacian_variance(image):
    """计算图像的拉普拉斯方差（清晰度指标）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    """计算图像平均亮度（曝光度指标）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return np.mean(gray)

def calculate_blur_score(image):
    """
    基于梯度幅值的标准差估算模糊度（值越小越模糊）
    参考：https://ieeexplore.ieee.org/document/5540096
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.std(magnitude) / np.mean(magnitude + 1e-6)  # 避免除零

def estimate_noise_level(image):
    """
    使用小波域方法粗略估计噪声水平（简化版）
    返回高噪声区域占比（>阈值的像素比例）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # 使用高斯差分近似高频噪声
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    diff = cv2.absdiff(blur1, blur2)
    # 设定噪声阈值（经验值）
    noise_mask = diff > 10
    return np.mean(noise_mask)

def is_valid_image(
    image_path,
    laplacian_threshold=85,
    brightness_low=30,
    brightness_high=220,
    blur_threshold=0.25,
    noise_ratio_threshold=0.12,
    verbose=False
):
    """
    判断图像是否通过质量筛选
    返回: (是否保留, 质量指标字典)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            if verbose:
                print(f"Warning: Cannot read image {image_path}")
            return False, {}

        lap_var = calculate_laplacian_variance(img)
        brightness = calculate_brightness(img)
        blur_score = calculate_blur_score(img)
        noise_ratio = estimate_noise_level(img)

        metrics = {
            "laplacian_variance": lap_var,
            "brightness": brightness,
            "blur_score": blur_score,
            "noise_ratio": noise_ratio
        }

        # 判断是否剔除
        reject_reasons = []
        if lap_var < laplacian_threshold:
            reject_reasons.append("low_sharpness")
        if brightness < brightness_low or brightness > brightness_high:
            reject_reasons.append("bad_exposure")
        if blur_score < blur_threshold:  # 注意：blur_score 越小越模糊
            reject_reasons.append("motion_or_defocus_blur")
        if noise_ratio > noise_ratio_threshold:
            reject_reasons.append("high_noise")

        keep = len(reject_reasons) == 0
        if verbose and not keep:
            print(f"{image_path.name}: rejected due to {reject_reasons}")

        return keep, metrics

    except Exception as e:
        if verbose:
            print(f"Error processing {image_path}: {e}")
        return False, {}

def main():
    parser = argparse.ArgumentParser(description="Batch quality check for NEODrone dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_csv", type=str, default="quality_report.csv", help="Output CSV file path")
    parser.add_argument("--ext", type=str, default=".jpg", help="Image file extension (e.g., .jpg, .png)")
    parser.add_argument("--laplacian_threshold", type=float, default=85.0)
    parser.add_argument("--brightness_low", type=int, default=30)
    parser.add_argument("--brightness_high", type=int, default=220)
    parser.add_argument("--blur_threshold", type=float, default=0.25)
    parser.add_argument("--noise_ratio_threshold", type=float, default=0.12)
    parser.add_argument("--verbose", action="store_true", help="Print rejection reasons")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_paths = list(input_dir.rglob(f"*{args.ext}"))
    if not image_paths:
        print(f"No images found with extension {args.ext} in {input_dir}")
        return

    results = []
    for img_path in sorted(image_paths):
        keep, metrics = is_valid_image(
            img_path,
            laplacian_threshold=args.laplacian_threshold,
            brightness_low=args.brightness_low,
            brightness_high=args.brightness_high,
            blur_threshold=args.blur_threshold,
            noise_ratio_threshold=args.noise_ratio_threshold,
            verbose=args.verbose
        )
        row = {"image_path": str(img_path), "keep": keep}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Quality check completed. Results saved to {args.output_csv}")
    print(f"Total images: {len(df)}")
    print(f"Images to keep: {df['keep'].sum()} ({df['keep'].mean()*100:.2f}%)")

if __name__ == "__main__":
    main()