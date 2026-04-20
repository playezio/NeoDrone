import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiModalSyncExtractor:
    def __init__(self, max_time_offset_ms: float = 10.0):
        self.max_time_offset_ms = max_time_offset_ms
        self.max_pixel_deviation = 5
        
    def parse_timestamp_from_filename(self, filename: str) -> datetime:
        try:
            parts = filename.replace('.jpg', '').replace('.png', '').split('_')
            timestamp_str = parts[-1]
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except Exception as e:
            logger.warning(f"Could not parse timestamp from {filename}: {e}")
            return None
    
    def calculate_time_offset(self, visible_frame: str, thermal_frame: str) -> float:
        visible_time = self.parse_timestamp_from_filename(visible_frame)
        thermal_time = self.parse_timestamp_from_filename(thermal_frame)
        
        if visible_time and thermal_time:
            offset = abs((visible_time - thermal_time).total_seconds() * 1000)
            return offset
        return float('inf')
    
    def find_matching_frames(self, visible_frames: List[str], thermal_frames: List[str]) -> List[Tuple[str, str]]:
        matched_pairs = []
        
        for visible_frame in visible_frames:
            best_match = None
            min_offset = float('inf')
            
            for thermal_frame in thermal_frames:
                offset = self.calculate_time_offset(visible_frame, thermal_frame)
                if offset < min_offset:
                    min_offset = offset
                    best_match = thermal_frame
            
            if best_match and min_offset <= self.max_time_offset_ms:
                matched_pairs.append((visible_frame, best_match))
                logger.debug(f"Matched: {visible_frame} <-> {best_match} (offset: {min_offset:.2f}ms)")
            else:
                logger.warning(f"No match found for {visible_frame} (min offset: {min_offset:.2f}ms)")
        
        return matched_pairs
    
    def align_frames_spatially(self, visible_path: str, thermal_path: str) -> Tuple[np.ndarray, np.ndarray]:
        visible = cv2.imread(visible_path)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        if visible is None or thermal is None:
            raise ValueError(f"Could not load frames: {visible_path} or {thermal_path}")
        
        visible_gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(visible_gray, None)
        kp2, des2 = orb.detectAndCompute(thermal, None)
        
        if des1 is None or des2 is None:
            logger.warning(f"Could not find features in {visible_path} or {thermal_path}")
            return visible, thermal
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:
            logger.warning(f"Insufficient matches ({len(matches)}) for {visible_path}")
            return visible, thermal
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            aligned_thermal = cv2.warpPerspective(thermal, M, (visible.shape[1], visible.shape[0]))
            return visible, aligned_thermal
        else:
            logger.warning(f"Could not compute homography for {visible_path}")
            return visible, thermal
    
    def validate_alignment(self, visible: np.ndarray, thermal: np.ndarray) -> bool:
        visible_gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(visible_gray, None)
        kp2, des2 = orb.detectAndCompute(thermal, None)
        
        if des1 is None or des2 is None:
            return False
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10:
            return False
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        
        deviations = np.linalg.norm(src_pts - dst_pts, axis=2)
        mean_deviation = np.mean(deviations)
        
        return mean_deviation <= self.max_pixel_deviation
    
    def process_synced_frames(self, visible_dir: str, thermal_dir: str, output_dir: str) -> Dict:
        visible_frames = sorted(list(Path(visible_dir).glob('*.jpg')))
        thermal_frames = sorted(list(Path(thermal_dir).glob('*.png')))
        
        logger.info(f"Found {len(visible_frames)} visible frames and {len(thermal_frames)} thermal frames")
        
        matched_pairs = self.find_matching_frames([str(f) for f in visible_frames], [str(f) for f in thermal_frames])
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'total_pairs': len(matched_pairs),
            'successful_alignments': 0,
            'failed_alignments': 0,
            'pairs': []
        }
        
        for visible_path, thermal_path in matched_pairs:
            try:
                visible, thermal_aligned = self.align_frames_spatially(visible_path, thermal_path)
                
                if self.validate_alignment(visible, thermal_aligned):
                    frame_name = Path(visible_path).stem
                    
                    visible_output = output_path / f"{frame_name}_visible.jpg"
                    thermal_output = output_path / f"{frame_name}_thermal.png"
                    
                    cv2.imwrite(str(visible_output), visible, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(str(thermal_output), thermal_aligned)
                    
                    results['successful_alignments'] += 1
                    results['pairs'].append({
                        'visible': str(visible_output),
                        'thermal': str(thermal_output),
                        'status': 'success'
                    })
                    
                    logger.info(f"Successfully aligned and saved: {frame_name}")
                else:
                    results['failed_alignments'] += 1
                    results['pairs'].append({
                        'visible': visible_path,
                        'thermal': thermal_path,
                        'status': 'failed_alignment'
                    })
                    logger.warning(f"Alignment validation failed for: {Path(visible_path).stem}")
                    
            except Exception as e:
                results['failed_alignments'] += 1
                results['pairs'].append({
                    'visible': visible_path,
                    'thermal': thermal_path,
                    'status': 'error',
                    'error': str(e)
                })
                logger.error(f"Error processing pair {visible_path} - {thermal_path}: {e}")
        
        return results
    
    def save_sync_report(self, results: Dict, output_path: str):
        report_path = Path(output_path) / 'sync_report.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Sync report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Synchronize visible and thermal frames')
    parser.add_argument('--visible', type=str, required=True, help='Directory containing visible frames')
    parser.add_argument('--thermal', type=str, required=True, help='Directory containing thermal frames')
    parser.add_argument('--output', type=str, required=True, help='Output directory for synced frames')
    parser.add_argument('--max-offset', type=float, default=10.0, help='Maximum time offset in milliseconds')
    args = parser.parse_args()
    
    sync_extractor = MultiModalSyncExtractor(max_time_offset_ms=args.max_offset)
    results = sync_extractor.process_synced_frames(args.visible, args.thermal, args.output)
    sync_extractor.save_sync_report(results, args.output)
    
    logger.info(f"Sync complete: {results['successful_alignments']}/{results['total_pairs']} pairs successfully aligned")


if __name__ == '__main__':
    main()