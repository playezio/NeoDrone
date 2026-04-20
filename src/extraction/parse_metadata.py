import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import exifread
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetadataParser:
    def __init__(self, vocab_path: str = None):
        self.metadata_fields = [
            'flight_id', 'timestamp', 'latitude', 'longitude', 'altitude',
            'heading', 'pitch', 'roll', 'gimbal_pitch', 'gimbal_roll',
            'camera_zoom', 'camera_focus', 'acquisition_weather', 'acquisition_scene',
            'sensor_type'
        ]
        
        self.vocab = {}
        if vocab_path:
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
    
    def parse_exif_data(self, image_path: str) -> Dict:
        exif_data = {}
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                if 'EXIF DateTimeOriginal' in tags:
                    exif_data['timestamp'] = str(tags['EXIF DateTimeOriginal'])
                
                if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                    lat = self._convert_to_degrees(tags['GPS GPSLatitude'])
                    lon = self._convert_to_degrees(tags['GPS GPSLongitude'])
                    exif_data['latitude'] = lat
                    exif_data['longitude'] = lon
                
                if 'EXIF FocalLength' in tags:
                    exif_data['camera_focus'] = float(str(tags['EXIF FocalLength']))
                
        except Exception as e:
            logger.warning(f"Could not parse EXIF from {image_path}: {e}")
        
        return exif_data
    
    def _convert_to_degrees(self, value) -> float:
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    def parse_sdk_log(self, log_path: str) -> Dict:
        sdk_data = {}
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                
                if 'telemetry' in log_data:
                    telemetry = log_data['telemetry']
                    sdk_data.update({
                        'latitude': telemetry.get('latitude'),
                        'longitude': telemetry.get('longitude'),
                        'altitude': telemetry.get('altitude'),
                        'heading': telemetry.get('heading'),
                        'pitch': telemetry.get('pitch'),
                        'roll': telemetry.get('roll')
                    })
                
                if 'camera' in log_data:
                    camera = log_data['camera']
                    sdk_data.update({
                        'gimbal_pitch': camera.get('gimbal_pitch'),
                        'gimbal_roll': camera.get('gimbal_roll'),
                        'camera_zoom': camera.get('zoom_level'),
                        'camera_focus': camera.get('focus_distance')
                    })
                    
        except Exception as e:
            logger.warning(f"Could not parse SDK log from {log_path}: {e}")
        
        return sdk_data
    
    def extract_flight_id(self, filename: str) -> str:
        parts = Path(filename).stem.split('_')
        return parts[0] if parts else 'unknown'
    
    def determine_sensor_type(self, filename: str) -> str:
        filename_lower = filename.lower()
        if 'thermal' in filename_lower or 'ir' in filename_lower:
            return 'thermal'
        elif 'visible' in filename_lower or 'rgb' in filename_lower:
            return 'visible'
        elif 'multispectral' in filename_lower or 'ms' in filename_lower:
            return 'multispectral'
        else:
            return 'unknown'
    
    def infer_weather_scene(self, image_path: str) -> Dict:
        import cv2
        import numpy as np
        
        weather_scene = {
            'acquisition_weather': 'unknown',
            'acquisition_scene': 'unknown'
        }
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return weather_scene
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness < 50:
                weather_scene['acquisition_weather'] = 'night'
            elif brightness < 100:
                weather_scene['acquisition_weather'] = 'cloudy'
            else:
                weather_scene['acquisition_weather'] = 'sunny'
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1:
                weather_scene['acquisition_scene'] = 'urban'
            elif edge_density > 0.05:
                weather_scene['acquisition_scene'] = 'suburban'
            else:
                weather_scene['acquisition_scene'] = 'rural'
                
        except Exception as e:
            logger.warning(f"Could not infer weather/scene from {image_path}: {e}")
        
        return weather_scene
    
    def merge_metadata(self, exif_data: Dict, sdk_data: Dict, filename: str, image_path: str) -> Dict:
        merged = {}
        
        for field in self.metadata_fields:
            if field in sdk_data:
                merged[field] = sdk_data[field]
            elif field in exif_data:
                merged[field] = exif_data[field]
            elif field == 'flight_id':
                merged[field] = self.extract_flight_id(filename)
            elif field == 'sensor_type':
                merged[field] = self.determine_sensor_type(filename)
        
        if 'acquisition_weather' not in merged or 'acquisition_scene' not in merged:
            inferred = self.infer_weather_scene(image_path)
            merged.update(inferred)
        
        return merged
    
    def process_images(self, image_dir: str, sdk_logs_dir: str) -> pd.DataFrame:
        image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        logger.info(f"Found {len(image_files)} images to process")
        
        metadata_records = []
        
        for image_file in image_files:
            filename = image_file.name
            logger.info(f"Processing: {filename}")
            
            exif_data = self.parse_exif_data(str(image_file))
            
            flight_id = self.extract_flight_id(filename)
            log_file = Path(sdk_logs_dir) / f"{flight_id}.json"
            
            sdk_data = {}
            if log_file.exists():
                sdk_data = self.parse_sdk_log(str(log_file))
            
            merged_metadata = self.merge_metadata(exif_data, sdk_data, filename, str(image_file))
            merged_metadata['image_path'] = str(image_file)
            
            metadata_records.append(merged_metadata)
        
        df = pd.DataFrame(metadata_records)
        
        for field in self.metadata_fields:
            if field not in df.columns:
                df[field] = None
        
        df = df[self.metadata_fields + ['image_path']]
        
        return df
    
    def save_metadata(self, df: pd.DataFrame, output_path: str):
        df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"Metadata saved to {output_path}")
    
    def validate_metadata(self, df: pd.DataFrame) -> Dict:
        validation_results = {
            'total_records': len(df),
            'complete_records': 0,
            'missing_fields': {},
            'invalid_values': {}
        }
        
        for field in self.metadata_fields:
            missing_count = df[field].isna().sum()
            if missing_count > 0:
                validation_results['missing_fields'][field] = missing_count
        
        complete_mask = df[self.metadata_fields].notna().all(axis=1)
        validation_results['complete_records'] = complete_mask.sum()
        
        return validation_results


def main():
    parser = argparse.ArgumentParser(description='Parse metadata from images and SDK logs')
    parser.add_argument('--sdk_logs', type=str, required=True, help='Directory containing SDK log files')
    parser.add_argument('--exif_data', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--output', type=str, required=True, help='Output Excel file path')
    parser.add_argument('--vocab', type=str, help='Path to vocabulary JSON file')
    args = parser.parse_args()
    
    parser_instance = MetadataParser(vocab_path=args.vocab)
    df = parser_instance.process_images(args.exif_data, args.sdk_logs)
    parser_instance.save_metadata(df, args.output)
    
    validation = parser_instance.validate_metadata(df)
    logger.info(f"Validation results: {validation}")
    
    logger.info(f"Metadata parsing complete: {validation['complete_records']}/{validation['total_records']} complete records")


if __name__ == '__main__':
    main()