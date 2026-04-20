import cv2
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlignmentChecker:
    def __init__(self, max_deviation: float = 5.0, min_overlap: float = 0.8):
        self.max_deviation = max_deviation
        self.min_overlap = min_overlap
    
    def find_matching_pairs(self, visible_dir: str, thermal_dir: str) -> List[Tuple[str, str]]:
        visible_files = sorted(list(Path(visible_dir).glob('*_visible.jpg')))
        thermal_files = sorted(list(Path(thermal_dir).glob('*_thermal.png')))
        
        pairs = []
        
        for visible_file in visible_files:
            base_name = visible_file.stem.replace('_visible', '')
            thermal_file = Path(thermal_dir) / f"{base_name}_thermal.png"
            
            if thermal_file.exists():
                pairs.append((str(visible_file), str(thermal_file)))
            else:
                logger.warning(f"No matching thermal file for {visible_file.name}")
        
        logger.info(f"Found {len(pairs)} matching pairs")
        return pairs
    
    def calculate_overlap(self, bbox1: Tuple[float, float, float, float], 
                         bbox2: Tuple[float, float, float, float]) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        intersection_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersection_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection_area = intersection_x * intersection_y
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def detect_keypoints(self, image: np.ndarray, max_features: int = 1000) -> Tuple[List, np.ndarray]:
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_keypoints(self, desc1: np.ndarray, desc2: np.ndarray, 
                       ratio_threshold: float = 0.75) -> List:
        if desc1 is None or desc2 is None:
            return []
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def calculate_homography(self, kp1: List, kp2: List, matches: List) -> Tuple[np.ndarray, np.ndarray]:
        if len(matches) < 4:
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return M, mask
    
    def calculate_alignment_deviation(self, visible_path: str, thermal_path: str) -> Dict[str, any]:
        visible = cv2.imread(visible_path)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        if visible is None or thermal is None:
            return {
                'status': 'error',
                'error': 'Could not load images',
                'visible_path': visible_path,
                'thermal_path': thermal_path
            }
        
        visible_gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
        
        kp1, desc1 = self.detect_keypoints(visible_gray)
        kp2, desc2 = self.detect_keypoints(thermal)
        
        if desc1 is None or desc2 is None:
            return {
                'status': 'error',
                'error': 'Could not detect keypoints',
                'visible_path': visible_path,
                'thermal_path': thermal_path,
                'keypoints_visible': len(kp1) if kp1 else 0,
                'keypoints_thermal': len(kp2) if kp2 else 0
            }
        
        matches = self.match_keypoints(desc1, desc2)
        
        if len(matches) < 10:
            return {
                'status': 'insufficient_matches',
                'matches': len(matches),
                'visible_path': visible_path,
                'thermal_path': thermal_path,
                'is_aligned': False
            }
        
        M, mask = self.calculate_homography(kp1, kp2, matches)
        
        if M is None:
            return {
                'status': 'homography_failed',
                'matches': len(matches),
                'visible_path': visible_path,
                'thermal_path': thermal_path,
                'is_aligned': False
            }
        
        inlier_matches = [m for i, m in enumerate(matches) if mask[i][0] == 1]
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        
        deviations = np.linalg.norm(src_pts - dst_pts, axis=2)
        mean_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        std_deviation = np.std(deviations)
        
        is_aligned = mean_deviation <= self.max_deviation
        
        return {
            'status': 'success',
            'visible_path': visible_path,
            'thermal_path': thermal_path,
            'matches': len(matches),
            'inliers': len(inlier_matches),
            'mean_deviation': float(mean_deviation),
            'max_deviation': float(max_deviation),
            'std_deviation': float(std_deviation),
            'is_aligned': is_aligned,
            'threshold': self.max_deviation
        }
    
    def calculate_bbox_alignment(self, visible_path: str, thermal_path: str, 
                                visible_bboxes: List, thermal_bboxes: List) -> Dict[str, any]:
        if not visible_bboxes or not thermal_bboxes:
            return {
                'status': 'no_bboxes',
                'visible_bboxes_count': len(visible_bboxes),
                'thermal_bboxes_count': len(thermal_bboxes)
            }
        
        alignments = []
        
        for vis_bbox in visible_bboxes:
            best_match = None
            best_overlap = 0.0
            
            for therm_bbox in thermal_bboxes:
                overlap = self.calculate_overlap(vis_bbox, therm_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = therm_bbox
            
            if best_match and best_overlap >= self.min_overlap:
                vis_center = ((vis_bbox[0] + vis_bbox[2]) / 2, (vis_bbox[1] + vis_bbox[3]) / 2)
                therm_center = ((best_match[0] + best_match[2]) / 2, (best_match[1] + best_match[3]) / 2)
                
                deviation = np.sqrt((vis_center[0] - therm_center[0])**2 + 
                                  (vis_center[1] - therm_center[1])**2)
                
                alignments.append({
                    'visible_bbox': vis_bbox,
                    'thermal_bbox': best_match,
                    'overlap': best_overlap,
                    'center_deviation': deviation
                })
        
        if not alignments:
            return {
                'status': 'no_alignments',
                'visible_bboxes_count': len(visible_bboxes),
                'thermal_bboxes_count': len(thermal_bboxes),
                'alignments_found': 0
            }
        
        mean_deviation = np.mean([a['center_deviation'] for a in alignments])
        max_deviation = np.max([a['center_deviation'] for a in alignments])
        mean_overlap = np.mean([a['overlap'] for a in alignments])
        
        is_aligned = mean_deviation <= self.max_deviation
        
        return {
            'status': 'success',
            'alignments_found': len(alignments),
            'mean_center_deviation': float(mean_deviation),
            'max_center_deviation': float(max_deviation),
            'mean_overlap': float(mean_overlap),
            'is_aligned': is_aligned,
            'threshold': self.max_deviation,
            'alignments': alignments
        }
    
    def process_directory(self, visible_dir: str, thermal_dir: str, output_report_path: str) -> Dict:
        pairs = self.find_matching_pairs(visible_dir, thermal_dir)
        
        results = {
            'total_pairs': len(pairs),
            'aligned': 0,
            'not_aligned': 0,
            'errors': 0,
            'pair_results': []
        }
        
        for visible_path, thermal_path in pairs:
            logger.info(f"Checking alignment: {Path(visible_path).name}")
            
            alignment_result = self.calculate_alignment_deviation(visible_path, thermal_path)
            results['pair_results'].append(alignment_result)
            
            if alignment_result['status'] == 'success':
                if alignment_result['is_aligned']:
                    results['aligned'] += 1
                    logger.info(f"Aligned: {Path(visible_path).name} (deviation: {alignment_result['mean_deviation']:.2f}px)")
                else:
                    results['not_aligned'] += 1
                    logger.warning(f"Not aligned: {Path(visible_path).name} (deviation: {alignment_result['mean_deviation']:.2f}px)")
            else:
                results['errors'] += 1
                logger.error(f"Error checking {Path(visible_path).name}: {alignment_result['status']}")
        
        with open(output_report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Alignment check complete. Report saved to {output_report_path}")
        logger.info(f"Summary: {results['aligned']}/{results['total_pairs']} pairs aligned")
        
        return results
    
    def generate_summary_report(self, results: Dict, output_csv_path: str):
        summary_data = []
        
        for pair_result in results['pair_results']:
            if pair_result['status'] == 'success':
                summary_data.append({
                    'visible_path': Path(pair_result['visible_path']).name,
                    'thermal_path': Path(pair_result['thermal_path']).name,
                    'matches': pair_result['matches'],
                    'inliers': pair_result['inliers'],
                    'mean_deviation': pair_result['mean_deviation'],
                    'max_deviation': pair_result['max_deviation'],
                    'is_aligned': pair_result['is_aligned'],
                    'threshold': pair_result['threshold']
                })
            else:
                summary_data.append({
                    'visible_path': Path(pair_result['visible_path']).name,
                    'thermal_path': Path(pair_result['thermal_path']).name,
                    'status': pair_result['status'],
                    'is_aligned': False
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Summary report saved to {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Check alignment between visible and thermal images')
    parser.add_argument('--visible', type=str, required=True, help='Directory containing visible images')
    parser.add_argument('--thermal', type=str, required=True, help='Directory containing thermal images')
    parser.add_argument('--output', type=str, required=True, help='Output JSON report path')
    parser.add_argument('--csv-output', type=str, help='Output CSV summary report path')
    parser.add_argument('--max-deviation', type=float, default=5.0, help='Maximum allowed deviation in pixels')
    parser.add_argument('--min-overlap', type=float, default=0.8, help='Minimum overlap ratio for bbox matching')
    args = parser.parse_args()
    
    checker = AlignmentChecker(max_deviation=args.max_deviation, min_overlap=args.min_overlap)
    results = checker.process_directory(args.visible, args.thermal, args.output)
    
    if args.csv_output:
        checker.generate_summary_report(results, args.csv_output)


if __name__ == '__main__':
    main()