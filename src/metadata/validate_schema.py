import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from jsonschema import validate, ValidationError
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetadataSchemaValidator:
    def __init__(self, schema_path: str = None, vocab_path: str = None):
        self.metadata_fields = [
            'flight_id', 'timestamp', 'latitude', 'longitude', 'altitude',
            'heading', 'pitch', 'roll', 'gimbal_pitch', 'gimbal_roll',
            'camera_zoom', 'camera_focus', 'acquisition_weather', 'acquisition_scene',
            'sensor_type'
        ]
        
        self.schema = self._load_schema(schema_path) if schema_path else self._default_schema()
        self.vocab = self._load_vocab(vocab_path) if vocab_path else {}
    
    def _default_schema(self) -> Dict:
        return {
            "type": "object",
            "required": self.metadata_fields,
            "properties": {
                "flight_id": {"type": "string", "minLength": 1},
                "timestamp": {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"},
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "altitude": {"type": "number", "minimum": 0},
                "heading": {"type": "number", "minimum": 0, "maximum": 360},
                "pitch": {"type": "number", "minimum": -90, "maximum": 90},
                "roll": {"type": "number", "minimum": -180, "maximum": 180},
                "gimbal_pitch": {"type": "number", "minimum": -90, "maximum": 90},
                "gimbal_roll": {"type": "number", "minimum": -180, "maximum": 180},
                "camera_zoom": {"type": "number", "minimum": 1},
                "camera_focus": {"type": "number", "minimum": 0},
                "acquisition_weather": {"type": "string"},
                "acquisition_scene": {"type": "string"},
                "sensor_type": {"type": "string", "enum": ["visible", "thermal", "multispectral"]}
            }
        }
    
    def _load_schema(self, schema_path: str) -> Dict:
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def _load_vocab(self, vocab_path: str) -> Dict:
        with open(vocab_path, 'r') as f:
            return json.load(f)
    
    def validate_field_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'total_records': len(df),
            'complete_records': 0,
            'missing_fields': {},
            'completeness_percentage': 0.0
        }
        
        for field in self.metadata_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                if missing_count > 0:
                    results['missing_fields'][field] = {
                        'missing_count': int(missing_count),
                        'missing_percentage': float(missing_count / len(df) * 100)
                    }
            else:
                results['missing_fields'][field] = {
                    'missing_count': len(df),
                    'missing_percentage': 100.0,
                    'note': 'Field not present in dataframe'
                }
        
        complete_mask = df[self.metadata_fields].notna().all(axis=1)
        results['complete_records'] = int(complete_mask.sum())
        results['completeness_percentage'] = float(results['complete_records'] / len(df) * 100) if len(df) > 0 else 0.0
        
        return results
    
    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'type_errors': {},
            'valid_records': len(df)
        }
        
        type_mapping = {
            'flight_id': 'object',
            'timestamp': 'object',
            'latitude': 'float64',
            'longitude': 'float64',
            'altitude': 'float64',
            'heading': 'float64',
            'pitch': 'float64',
            'roll': 'float64',
            'gimbal_pitch': 'float64',
            'gimbal_roll': 'float64',
            'camera_zoom': 'float64',
            'camera_focus': 'float64',
            'acquisition_weather': 'object',
            'acquisition_scene': 'object',
            'sensor_type': 'object'
        }
        
        for field, expected_type in type_mapping.items():
            if field in df.columns:
                try:
                    if expected_type == 'float64':
                        pd.to_numeric(df[field], errors='coerce')
                    elif expected_type == 'object':
                        df[field].astype(str)
                except Exception as e:
                    results['type_errors'][field] = str(e)
                    results['valid_records'] -= df[field].isna().sum()
        
        return results
    
    def validate_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'range_violations': {},
            'outlier_count': 0
        }
        
        range_constraints = {
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'altitude': (0, None),
            'heading': (0, 360),
            'pitch': (-90, 90),
            'roll': (-180, 180),
            'gimbal_pitch': (-90, 90),
            'gimbal_roll': (-180, 180),
            'camera_zoom': (1, None),
            'camera_focus': (0, None)
        }
        
        for field, (min_val, max_val) in range_constraints.items():
            if field in df.columns:
                numeric_values = pd.to_numeric(df[field], errors='coerce')
                
                if min_val is not None:
                    violations_min = (numeric_values < min_val).sum()
                    if violations_min > 0:
                        results['range_violations'][f'{field}_below_min'] = int(violations_min)
                        results['outlier_count'] += violations_min
                
                if max_val is not None:
                    violations_max = (numeric_values > max_val).sum()
                    if violations_max > 0:
                        results['range_violations'][f'{field}_above_max'] = int(violations_max)
                        results['outlier_count'] += violations_max
        
        return results
    
    def validate_vocabulary(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'vocab_violations': {},
            'invalid_values': {}
        }
        
        vocab_fields = ['acquisition_weather', 'acquisition_scene', 'sensor_type']
        
        for field in vocab_fields:
            if field in df.columns and field in self.vocab:
                valid_values = set(self.vocab[field])
                unique_values = set(df[field].dropna().unique())
                invalid_values = unique_values - valid_values
                
                if invalid_values:
                    results['vocab_violations'][field] = len(invalid_values)
                    results['invalid_values'][field] = list(invalid_values)
        
        return results
    
    def validate_schema_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'compliant_records': 0,
            'non_compliant_records': 0,
            'validation_errors': []
        }
        
        for idx, row in df.iterrows():
            record = row.to_dict()
            
            try:
                validate(instance=record, schema=self.schema)
                results['compliant_records'] += 1
            except ValidationError as e:
                results['non_compliant_records'] += 1
                results['validation_errors'].append({
                    'row_index': idx,
                    'field': e.path[-1] if e.path else 'unknown',
                    'error': e.message
                })
        
        return results
    
    def validate_timestamp_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'invalid_timestamps': 0,
            'invalid_timestamp_records': []
        }
        
        if 'timestamp' in df.columns:
            for idx, timestamp in df['timestamp'].items():
                if pd.notna(timestamp):
                    try:
                        pd.to_datetime(timestamp)
                    except Exception as e:
                        results['invalid_timestamps'] += 1
                        results['invalid_timestamp_records'].append({
                            'row_index': idx,
                            'timestamp': str(timestamp),
                            'error': str(e)
                        })
        
        return results
    
    def validate_image_paths(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'missing_images': 0,
            'invalid_paths': []
        }
        
        if 'image_path' in df.columns:
            for idx, image_path in df['image_path'].items():
                if pd.notna(image_path):
                    path = Path(image_path)
                    if not path.exists():
                        results['missing_images'] += 1
                        results['invalid_paths'].append({
                            'row_index': idx,
                            'image_path': str(image_path)
                        })
        
        return results
    
    def comprehensive_validation(self, metadata_path: str) -> Dict[str, Any]:
        logger.info(f"Loading metadata from {metadata_path}")
        df = pd.read_excel(metadata_path)
        
        logger.info("Starting comprehensive validation...")
        
        validation_results = {
            'metadata_file': metadata_path,
            'total_records': len(df),
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'field_completeness': self.validate_field_completeness(df),
            'data_types': self.validate_data_types(df),
            'value_ranges': self.validate_value_ranges(df),
            'vocabulary': self.validate_vocabulary(df),
            'schema_compliance': self.validate_schema_compliance(df),
            'timestamp_format': self.validate_timestamp_format(df),
            'image_paths': self.validate_image_paths(df),
            'overall_status': 'passed'
        }
        
        total_issues = (
            validation_results['field_completeness']['complete_records'] < len(df) or
            validation_results['data_types']['valid_records'] < len(df) or
            validation_results['value_ranges']['outlier_count'] > 0 or
            validation_results['vocabulary']['vocab_violations'] or
            validation_results['schema_compliance']['non_compliant_records'] > 0 or
            validation_results['timestamp_format']['invalid_timestamps'] > 0 or
            validation_results['image_paths']['missing_images'] > 0
        )
        
        if total_issues:
            validation_results['overall_status'] = 'failed'
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict, output_path: str):
        report_lines = [
            "Metadata Validation Report",
            "=" * 50,
            f"Metadata File: {validation_results['metadata_file']}",
            f"Validation Timestamp: {validation_results['validation_timestamp']}",
            f"Total Records: {validation_results['total_records']}",
            f"Overall Status: {validation_results['overall_status'].upper()}",
            "",
            "Field Completeness:",
            f"  Complete Records: {validation_results['field_completeness']['complete_records']}",
            f"  Completeness: {validation_results['field_completeness']['completeness_percentage']:.2f}%"
        ]
        
        if validation_results['field_completeness']['missing_fields']:
            report_lines.append("  Missing Fields:")
            for field, info in validation_results['field_completeness']['missing_fields'].items():
                report_lines.append(f"    {field}: {info['missing_count']} missing ({info['missing_percentage']:.2f}%)")
        
        report_lines.extend([
            "",
            "Schema Compliance:",
            f"  Compliant Records: {validation_results['schema_compliance']['compliant_records']}",
            f"  Non-Compliant Records: {validation_results['schema_compliance']['non_compliant_records']}"
        ])
        
        if validation_results['schema_compliance']['validation_errors']:
            report_lines.append("  Sample Errors (first 5):")
            for error in validation_results['schema_compliance']['validation_errors'][:5]:
                report_lines.append(f"    Row {error['row_index']}: {error['field']} - {error['error']}")
        
        report_lines.extend([
            "",
            "Value Range Violations:",
            f"  Total Outliers: {validation_results['value_ranges']['outlier_count']}"
        ])
        
        if validation_results['value_ranges']['range_violations']:
            report_lines.append("  Violations by Field:")
            for violation, count in validation_results['value_ranges']['range_violations'].items():
                report_lines.append(f"    {violation}: {count}")
        
        report_content = '\n'.join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Validation report saved to {output_path}")
        
        return report_content


def main():
    parser = argparse.ArgumentParser(description='Validate metadata schema and quality')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata Excel file')
    parser.add_argument('--schema', type=str, help='Path to JSON schema file')
    parser.add_argument('--vocab', type=str, help='Path to vocabulary JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for validation reports')
    args = parser.parse_args()
    
    validator = MetadataSchemaValidator(schema_path=args.schema, vocab_path=args.vocab)
    validation_results = validator.comprehensive_validation(args.metadata)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / 'validation_report.txt'
    validator.generate_validation_report(validation_results, str(report_path))
    
    json_path = output_path / 'validation_results.json'
    with open(json_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    logger.info(f"Validation complete! Status: {validation_results['overall_status']}")


if __name__ == '__main__':
    main()