import cv2
import numpy as np
import argparse
import yaml
import os
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from ultralytics import YOLO
from skimage import measure, filters
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detection_config = self.config['detection']
        self.thermal_config = self.config['thermal']
        self.alignment_config = self.config['alignment']
        
        self.detection_model = None
        if self.detection_config['model']:
            logger.info(f"Loading detection model: {self.detection_config['model']}")
            self.detection_model = YOLO(self.detection_config['model'])
    
    def calculate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 / (1.0 + blur_score / 1000.0)
        
        brightness = np.mean(gray) / 255.0
        
        contrast = np.std(gray) / 255.0
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'sharpness': laplacian_var,
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density
        }
    
    def is_quality_acceptable(self, quality_metrics: Dict[str, float]) -> bool:
        min_sharpness = self.config['quality'].get('min_sharpness', 100.0)
        max_blur_score = self.config['quality'].get('max_blur_score', 0.5)
        min_brightness = self.config['quality'].get('min_brightness', 0.1)
        max_brightness = self.config['quality'].get('max_brightness', 0.9)
        min_contrast = self.config['quality'].get('min_contrast', 0.05)
        min_edge_density = self.config['quality'].get('min_edge_density', 0.01)
        
        if quality_metrics['sharpness'] < min_sharpness:
            return False
        if quality_metrics['blur_score'] > max_blur_score:
            return False
        if not (min_brightness <= quality_metrics['brightness'] <= max_brightness):
            return False
        if quality_metrics['contrast'] < min_contrast:
            return False
        if quality_metrics['edge_density'] < min_edge_density:
            return False
        
        return True
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        if self.detection_model is None:
            return []
        
        results = self.detection_model(image, conf=self.detection_config['confidence_threshold'])
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id
                })
        
        return detections
    
    def has_sufficient_objects(self, detections: List[Dict], min_objects: int = 1) -> bool:
        return len(detections) >= min_objects
    
    def detect_thermal_anomalies(self, thermal_image: np.ndarray) -> Dict[str, any]:
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        anomalies = {
            'has_striping': False,
            'has_dead_pixels': False,
            'has_temperature_drift': False,
            'anomaly_percentage': 0.0
        }
        
        std_dev = np.std(thermal_gray)
        mean_val = np.mean(thermal_gray)
        
        threshold = mean_val + 3 * std_dev
        anomaly_mask = thermal_gray > threshold
        anomaly_count = np.sum(anomaly_mask)
        total_pixels = thermal_gray.size
        anomaly_percentage = anomaly_count / total_pixels
        
        anomalies['anomaly_percentage'] = anomaly_percentage
        
        if anomaly_percentage > self.thermal_config['anomaly_threshold']:
            anomalies['has_temperature_drift'] = True
        
        rows_mean = np.mean(thermal_gray, axis=1)
        rows_std = np.std(rows_mean)
        if rows_std > std_dev * 0.5:
            anomalies['has_striping'] = True
        
        dead_pixel_threshold = self.thermal_config['dead_pixel_threshold']
        min_val = np.min(thermal_gray)
        max_val = np.max(thermal_gray)
        
        if min_val < dead_pixel_threshold or max_val > (255 - dead_pixel_threshold):
            anomalies['has_dead_pixels'] = True
        
        return anomalies
    
    def is_thermal_acceptable(self, thermal_image: np.ndarray) -> bool:
        anomalies = self.detect_thermal_anomalies(thermal_image)
        
        if anomalies['has_temperature_drift']:
            logger.warning("Thermal image rejected: temperature drift detected")
            return False
        
        if anomalies['has_striping']:
            logger.warning("Thermal image rejected: striping noise detected")
            return False
        
        if anomalies['has_dead_pixels']:
            logger.warning("Thermal image rejected: dead pixels detected")
            return False
        
        return True
    
    def check_alignment(self, visible_path: str, thermal_path: str) -> bool:
        visible = cv2.imread(visible_path)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        if visible is None or thermal is None:
            return False
        
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
        
        matches = sorted(matches, key=lambda x: x.distance)[:20]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        deviations = np.linalg.norm(src_pts - dst_pts, axis=2)
        mean_deviation = np.mean(deviations)
        
        return mean_deviation <= self.alignment_config['max_deviation']
    
    def clean_image_pair(self, visible_path: str, thermal_path: str, output_dir: str) -> Dict[str, any]:
        result = {
            'visible_path': visible_path,
            'thermal_path': thermal_path,
            'status': 'rejected',
            'reason': '',
            'quality_metrics': {},
            'detections': [],
            'thermal_anomalies': {}
        }
        
        try:
            visible = cv2.imread(visible_path)
            thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            
            if visible is None or thermal is None:
                result['reason'] = 'Could not load images'
                return result
            
            quality_metrics = self.calculate_image_quality(visible)
            result['quality_metrics'] = quality_metrics
            
            if not self.is_quality_acceptable(quality_metrics):
                result['reason'] = 'Quality metrics not acceptable'
                return result
            
            detections = self.detect_objects(visible)
            result['detections'] = detections
            
            min_objects = self.detection_config.get('min_objects', 1)
            if not self.has_sufficient_objects(detections, min_objects):
                result['reason'] = 'Insufficient objects detected'
                return result
            
            thermal_anomalies = self.detect_thermal_anomalies(thermal)
            result['thermal_anomalies'] = thermal_anomalies
            
            if not self.is_thermal_acceptable(thermal):
                result['reason'] = 'Thermal anomalies detected'
                return result
            
            if not self.check_alignment(visible_path, thermal_path):
                result['reason'] = 'Alignment check failed'
                return result
            
            output_visible = Path(output_dir) / Path(visible_path).name
            output_thermal = Path(output_dir) / Path(thermal_path).name
            
            shutil.copy2(visible_path, output_visible)
            shutil.copy2(thermal_path, output_thermal)
            
            result['status'] = 'accepted'
            result['output_visible'] = str(output_visible)
            result['output_thermal'] = str(output_thermal)
            
        except Exception as e:
            result['reason'] = f'Error: {str(e)}'
            logger.error(f"Error processing pair {visible_path} - {thermal_path}: {e}")
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict:
        visible_files = sorted(list(Path(input_dir).glob('*_visible.jpg')))
        thermal_files = sorted(list(Path(input_dir).glob('*_thermal.png')))
        
        logger.info(f"Found {len(visible_files)} visible and {len(thermal_files)} thermal files")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'total_pairs': 0,
            'accepted': 0,
            'rejected': 0,
            'details': []
        }
        
        for visible_file in visible_files:
            base_name = visible_file.stem.replace('_visible', '')
            thermal_file = Path(input_dir) / f"{base_name}_thermal.png"
            
            if not thermal_file.exists():
                logger.warning(f"No matching thermal file for {visible_file.name}")
                continue
            
            results['total_pairs'] += 1
            
            result = self.clean_image_pair(str(visible_file), str(thermal_file), output_dir)
            results['details'].append(result)
            
            if result['status'] == 'accepted':
                results['accepted'] += 1
                logger.info(f"Accepted: {visible_file.name}")
            else:
                results['rejected'] += 1
                logger.info(f"Rejected: {visible_file.name} - {result['reason']}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Clean and filter image pairs')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing image pairs')
    parser.add_argument('--output', type=str, required=True, help='Output directory for cleaned images')
    parser.add_argument('--config', type=str, default='configs/cleaning_thresholds.yaml', help='Configuration file path')
    args = parser.parse_args()
    
    pipeline = DataCleaningPipeline(args.config)
    results = pipeline.process_directory(args.input, args.output)
    
    logger.info(f"Cleaning complete: {results['accepted']}/{results['total_pairs']} pairs accepted")
    logger.info(f"Rejected: {results['rejected']}/{results['total_pairs']} pairs")


if __name__ == '__main__':
    main()