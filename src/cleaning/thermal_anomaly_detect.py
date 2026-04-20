import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from scipy import ndimage
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThermalAnomalyDetector:
    def __init__(self, anomaly_threshold: float = 0.05, dead_pixel_threshold: int = 100):
        self.anomaly_threshold = anomaly_threshold
        self.dead_pixel_threshold = dead_pixel_threshold
    
    def detect_striping_noise(self, thermal_image: np.ndarray) -> Dict[str, any]:
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        rows_mean = np.mean(thermal_gray, axis=1)
        rows_std = np.std(rows_mean)
        global_std = np.std(thermal_gray)
        
        striping_ratio = rows_std / global_std if global_std > 0 else 0
        
        has_striping = striping_ratio > 0.3
        
        return {
            'has_striping': has_striping,
            'striping_ratio': float(striping_ratio),
            'rows_std': float(rows_std),
            'global_std': float(global_std)
        }
    
    def detect_dead_pixels(self, thermal_image: np.ndarray) -> Dict[str, any]:
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        min_val = np.min(thermal_gray)
        max_val = np.max(thermal_gray)
        mean_val = np.mean(thermal_gray)
        std_val = np.std(thermal_gray)
        
        dead_pixel_low = thermal_gray < self.dead_pixel_threshold
        dead_pixel_high = thermal_gray > (255 - self.dead_pixel_threshold)
        
        dead_pixel_count = np.sum(dead_pixel_low) + np.sum(dead_pixel_high)
        dead_pixel_percentage = dead_pixel_count / thermal_gray.size
        
        has_dead_pixels = dead_pixel_percentage > 0.01
        
        return {
            'has_dead_pixels': has_dead_pixels,
            'dead_pixel_percentage': float(dead_pixel_percentage),
            'dead_pixel_count': int(dead_pixel_count),
            'min_value': float(min_val),
            'max_value': float(max_val),
            'mean_value': float(mean_val),
            'std_value': float(std_val)
        }
    
    def detect_temperature_drift(self, thermal_image: np.ndarray) -> Dict[str, any]:
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        mean_val = np.mean(thermal_gray)
        std_val = np.std(thermal_gray)
        
        threshold_upper = mean_val + 3 * std_val
        threshold_lower = mean_val - 3 * std_val
        
        anomaly_mask = (thermal_gray > threshold_upper) | (thermal_gray < threshold_lower)
        anomaly_count = np.sum(anomaly_mask)
        total_pixels = thermal_gray.size
        anomaly_percentage = anomaly_count / total_pixels
        
        has_drift = anomaly_percentage > self.anomaly_threshold
        
        return {
            'has_temperature_drift': has_drift,
            'anomaly_percentage': float(anomaly_percentage),
            'anomaly_count': int(anomaly_count),
            'mean_value': float(mean_val),
            'std_value': float(std_val),
            'threshold_upper': float(threshold_upper),
            'threshold_lower': float(threshold_lower)
        }
    
    def detect_hotspots(self, thermal_image: np.ndarray) -> Dict[str, any]:
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        blurred = cv2.GaussianBlur(thermal_gray, (5, 5), 0)
        
        threshold = np.mean(blurred) + 2 * np.std(blurred)
        hotspot_mask = blurred > threshold
        
        labeled, num_features = ndimage.label(hotspot_mask)
        
        hotspots = []
        for i in range(1, num_features + 1):
            hotspot_region = (labeled == i)
            hotspot_pixels = np.sum(hotspot_region)
            hotspot_mean = np.mean(thermal_gray[hotspot_region])
            
            if hotspot_pixels > 10:
                hotspots.append({
                    'id': i,
                    'pixel_count': int(hotspot_pixels),
                    'mean_temperature': float(hotspot_mean)
                })
        
        return {
            'hotspot_count': len(hotspots),
            'hotspots': hotspots,
            'threshold': float(threshold)
        }
    
    def detect_fixed_pattern_noise(self, thermal_image: np.ndarray) -> Dict[str, any]:
        if len(thermal_image.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        fft = np.fft.fft2(thermal_gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        
        high_freq_energy = np.sum(magnitude_spectrum[center_y-50:center_y+50, center_x-50:center_x+50])
        total_energy = np.sum(magnitude_spectrum)
        
        pattern_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        has_pattern_noise = pattern_ratio > 0.1
        
        return {
            'has_pattern_noise': has_pattern_noise,
            'pattern_ratio': float(pattern_ratio),
            'high_freq_energy': float(high_freq_energy),
            'total_energy': float(total_energy)
        }
    
    def comprehensive_analysis(self, thermal_image_path: str) -> Dict[str, any]:
        thermal_image = cv2.imread(thermal_image_path, cv2.IMREAD_GRAYSCALE)
        
        if thermal_image is None:
            logger.error(f"Could not load thermal image: {thermal_image_path}")
            return {
                'error': 'Could not load image',
                'image_path': thermal_image_path
            }
        
        striping_result = self.detect_striping_noise(thermal_image)
        dead_pixel_result = self.detect_dead_pixels(thermal_image)
        drift_result = self.detect_temperature_drift(thermal_image)
        hotspot_result = self.detect_hotspots(thermal_image)
        pattern_result = self.detect_fixed_pattern_noise(thermal_image)
        
        overall_status = 'acceptable'
        issues = []
        
        if striping_result['has_striping']:
            overall_status = 'defective'
            issues.append('striping_noise')
        
        if dead_pixel_result['has_dead_pixels']:
            overall_status = 'defective'
            issues.append('dead_pixels')
        
        if drift_result['has_temperature_drift']:
            overall_status = 'defective'
            issues.append('temperature_drift')
        
        if pattern_result['has_pattern_noise']:
            overall_status = 'warning'
            issues.append('pattern_noise')
        
        return {
            'image_path': thermal_image_path,
            'overall_status': overall_status,
            'issues': issues,
            'striping_analysis': striping_result,
            'dead_pixel_analysis': dead_pixel_result,
            'temperature_drift_analysis': drift_result,
            'hotspot_analysis': hotspot_result,
            'pattern_noise_analysis': pattern_result
        }
    
    def process_directory(self, input_dir: str, output_report_path: str) -> Dict:
        thermal_files = list(Path(input_dir).glob('*.png')) + list(Path(input_dir).glob('*.jpg'))
        logger.info(f"Found {len(thermal_files)} thermal images to analyze")
        
        results = {
            'total_images': len(thermal_files),
            'acceptable': 0,
            'warning': 0,
            'defective': 0,
            'analyses': []
        }
        
        for thermal_file in thermal_files:
            logger.info(f"Analyzing: {thermal_file.name}")
            analysis = self.comprehensive_analysis(str(thermal_file))
            results['analyses'].append(analysis)
            
            status = analysis.get('overall_status', 'unknown')
            results[status] += 1
            
            if status == 'defective':
                logger.warning(f"Defective image found: {thermal_file.name} - Issues: {analysis['issues']}")
            elif status == 'warning':
                logger.info(f"Warning for image: {thermal_file.name} - Issues: {analysis['issues']}")
        
        with open(output_report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis complete. Report saved to {output_report_path}")
        logger.info(f"Summary: {results['acceptable']} acceptable, {results['warning']} warnings, {results['defective']} defective")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Detect anomalies in thermal images')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing thermal images')
    parser.add_argument('--output', type=str, required=True, help='Output JSON report path')
    parser.add_argument('--anomaly-threshold', type=float, default=0.05, help='Anomaly percentage threshold (0-1)')
    parser.add_argument('--dead-pixel-threshold', type=int, default=100, help='Dead pixel threshold (0-255)')
    args = parser.parse_args()
    
    detector = ThermalAnomalyDetector(
        anomaly_threshold=args.anomaly_threshold,
        dead_pixel_threshold=args.dead_pixel_threshold
    )
    
    results = detector.process_directory(args.input, args.output)


if __name__ == '__main__':
    main()