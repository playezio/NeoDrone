# img_cut.py

import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from math import tan, radians
from typing import Dict, List

def estimate_ground_coverage(height: float, fov_horizontal_deg: float = 70.0) -> float:
    """
    根据飞行高度和水平视场角，估算单帧图像在地面的覆盖宽度（单位：米）
    :param height: 飞行高度（AGL，单位：米）
    :param fov_horizontal_deg: 相机水平视场角（默认 70°，由 DFOV=84° 推算）
    :return: 地面覆盖宽度 W（米）
    """
    return 2 * height * tan(radians(fov_horizontal_deg / 2))

def adaptive_frame_sampling(
    video_path: Path,
    output_dir: Path,
    flight_logs: List[Dict],
    target_overlap: float = 0.3,
    fov_horizontal_deg: float = 70.0,
    min_interval_sec: float = 0.1,
    max_interval_sec: float = 2.0
):
    """
    根据飞行日志（含时间戳、高度、速度）自适应抽帧
    :param video_path: 输入视频路径
    :param output_dir: 输出图像目录
    :param flight_logs: 飞行日志列表，每个元素含 'timestamp', 'height', 'speed'
    :param target_overlap: 目标最大重叠率（如 0.3 表示 30%）
    :param fov_horizontal_deg: 水平视场角
    :param min_interval_sec: 最小抽帧间隔（秒）
    :param max_interval_sec: 最大抽帧间隔（秒）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")

    # 构建时间戳到帧号的映射（假设视频起始时间 = 第一条日志时间）
    start_time = flight_logs[0]['timestamp']  # 单位：秒
    frame_timestamps = [start_time + i / fps for i in range(total_frames)]

    # 按时间对齐飞行日志（简单线性插值）
    log_times = np.array([log['timestamp'] for log in flight_logs])
    log_heights = np.array([log['height'] for log in flight_logs])
    log_speeds = np.array([log['speed'] for log in flight_logs])

    output_dir.mkdir(parents=True, exist_ok=True)
    last_save_time = -1e9  # 上一次保存帧的时间戳
    saved_count = 0

    for frame_idx in range(total_frames):
        t = frame_timestamps[frame_idx]

        # 插值得到当前帧的高度和速度
        if t < log_times[0] or t > log_times[-1]:
            continue
        height = np.interp(t, log_times, log_heights)
        speed = np.interp(t, log_times, log_speeds)

        # 若速度接近0（悬停），按固定间隔抽帧（避免无限重叠）
        if speed < 0.1:
            interval_sec = max_interval_sec
        else:
            W = estimate_ground_coverage(height, fov_horizontal_deg)
            min_displacement = (1 - target_overlap) * W  # 如 overlap<0.3 → displacement > 0.7W
            interval_sec = min_displacement / speed
            # 限制在合理范围内
            interval_sec = np.clip(interval_sec, min_interval_sec, max_interval_sec)

        # 判断是否满足时间间隔
        if t - last_save_time >= interval_sec:
            ret, frame = cap.read()
            if not ret:
                break
            img_name = f"{video_path.stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / img_name), frame)
            saved_count += 1
            last_save_time = t
        else:
            # 跳过该帧
            cap.grab()  # 快速跳过，不 decode

    cap.release()
    print(f"Saved {saved_count} keyframes to {output_dir}")

def load_flight_log(log_path: Path) -> List[Dict]:
    """
    加载飞行日志（JSON格式），需包含字段：
    - timestamp: 视频起始对齐后的时间（秒）
    - height: 飞行高度（米）
    - speed: 水平速度（m/s）
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    # 确保按时间排序
    logs.sort(key=lambda x: x['timestamp'])
    return logs

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive keyframe extraction based on drone speed & height to control inter-frame overlap < 30%."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video file (.mp4/.avi)")
    parser.add_argument("--log", type=str, required=True, help="Flight log in JSON format with timestamp, height, speed")
    parser.add_argument("--output_dir", type=str, default="keyframes", help="Output directory for extracted images")
    parser.add_argument("--overlap", type=float, default=0.3, help="Max allowed overlap ratio (default: 0.3)")
    parser.add_argument("--fov_h", type=float, default=70.0, help="Camera horizontal FOV in degrees (default: 70.0)")

    args = parser.parse_args()

    flight_logs = load_flight_log(Path(args.log))
    adaptive_frame_sampling(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        flight_logs=flight_logs,
        target_overlap=args.overlap,
        fov_horizontal_deg=args.fov_h
    )

if __name__ == "__main__":
    main()