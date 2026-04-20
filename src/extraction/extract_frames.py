import cv2
import numpy as np
import argparse
import yaml
import os
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveFrameExtractor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sampling_strategy = self.config['sampling']['strategy']
        self.interval = self.config['sampling']['interval']
        self.overlap_threshold = self.config['sampling']['overlap_threshold']
        self.min_resolution = tuple(self.config['quality']['min_resolution'])
        self.min_sharpness = self.config['quality']['min_sharpness']
        self.max_blur_score = self.config['quality']['max_blur_score']
        
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0
    
    def calculate_sharpness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def is_scene_change(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        diff_score = self.calculate_frame_difference(prev_frame, curr_frame)
        return diff_score > self.overlap_threshold
    
    def is_quality_acceptable(self, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        if (width, height) < self.min_resolution:
            return False
        
        sharpness = self.calculate_sharpness(frame)
        if sharpness < self.min_sharpness:
            return False
        
        return True
    
    def extract_frames_adaptive(self, video_path: str, output_dir: str) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        extracted_frames = []
        prev_frame = None
        frame_count = 0
        saved_count = 0
        
        video_name = Path(video_path).stem
        output_path = Path(output_dir) / video_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing video: {video_name} (FPS: {fps}, Total frames: {total_frames})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if not self.is_quality_acceptable(frame):
                continue
            
            should_save = False
            
            if self.sampling_strategy == 'adaptive':
                if prev_frame is None:
                    should_save = True
                elif self.is_scene_change(prev_frame, frame):
                    should_save = True
            elif self.sampling_strategy == 'fixed':
                if frame_count % int(fps * self.interval) == 0:
                    should_save = True
            elif self.sampling_strategy == 'scene_change':
                if prev_frame is None or self.is_scene_change(prev_frame, frame):
                    should_save = True
            
            if should_save:
                frame_filename = f"{video_name}_{saved_count:06d}.jpg"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_frames.append(str(frame_path))
                saved_count += 1
                logger.info(f"Saved frame {saved_count}: {frame_filename}")
            
            prev_frame = frame.copy()
        
        cap.release()
        logger.info(f"Extracted {saved_count} frames from {video_name}")
        return extracted_frames
    
    def process_video_directory(self, input_dir: str, output_dir: str) -> dict:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(input_dir).glob(f'*{ext}'))
        
        results = {}
        for video_file in video_files:
            logger.info(f"Processing: {video_file.name}")
            try:
                extracted = self.extract_frames_adaptive(str(video_file), output_dir)
                results[video_file.name] = {
                    'status': 'success',
                    'frames_extracted': len(extracted),
                    'frames': extracted
                }
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
                results[video_file.name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos using adaptive sampling')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing videos')
    parser.add_argument('--output', type=str, required=True, help='Output directory for extracted frames')
    parser.add_argument('--config', type=str, default='configs/extraction_config.yaml', help='Configuration file path')
    args = parser.parse_args()
    
    extractor = AdaptiveFrameExtractor(args.config)
    results = extractor.process_video_directory(args.input, args.output)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_frames = sum(r.get('frames_extracted', 0) for r in results.values() if r['status'] == 'success')
    
    logger.info(f"Extraction complete: {success_count}/{len(results)} videos processed successfully")
    logger.info(f"Total frames extracted: {total_frames}")


if __name__ == '__main__':
    main()