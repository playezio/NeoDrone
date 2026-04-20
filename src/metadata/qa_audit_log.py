import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import json
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAAuditLogger:
    def __init__(self, sample_percentage: float = 5.0):
        self.sample_percentage = sample_percentage
        self.audit_fields = [
            'flight_id', 'timestamp', 'latitude', 'longitude', 'altitude',
            'heading', 'pitch', 'roll', 'gimbal_pitch', 'gimbal_roll',
            'camera_zoom', 'camera_focus', 'acquisition_weather', 'acquisition_scene',
            'sensor_type'
        ]
    
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        logger.info(f"Loading metadata from {metadata_path}")
        df = pd.read_excel(metadata_path)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def select_audit_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        sample_size = max(1, int(len(df) * self.sample_percentage / 100))
        logger.info(f"Selecting {sample_size} samples ({self.sample_percentage}%) for audit")
        
        sampled_df = df.sample(n=sample_size, random_state=42)
        return sampled_df
    
    def check_field_completeness(self, record: pd.Series) -> Dict[str, Any]:
        completeness = {
            'total_fields': len(self.audit_fields),
            'complete_fields': 0,
            'missing_fields': [],
            'completeness_score': 0.0
        }
        
        for field in self.audit_fields:
            if pd.notna(record.get(field, None)):
                completeness['complete_fields'] += 1
            else:
                completeness['missing_fields'].append(field)
        
        completeness['completeness_score'] = completeness['complete_fields'] / completeness['total_fields']
        
        return completeness
    
    def check_value_consistency(self, record: pd.Series) -> Dict[str, Any]:
        consistency = {
            'inconsistencies': [],
            'consistency_score': 1.0
        }
        
        if pd.notna(record.get('latitude')) and pd.notna(record.get('longitude')):
            if not (-90 <= record['latitude'] <= 90):
                consistency['inconsistencies'].append('Latitude out of valid range')
                consistency['consistency_score'] -= 0.1
            
            if not (-180 <= record['longitude'] <= 180):
                consistency['inconsistencies'].append('Longitude out of valid range')
                consistency['consistency_score'] -= 0.1
        
        if pd.notna(record.get('altitude')):
            if record['altitude'] < 0:
                consistency['inconsistencies'].append('Altitude cannot be negative')
                consistency['consistency_score'] -= 0.1
        
        if pd.notna(record.get('heading')):
            if not (0 <= record['heading'] <= 360):
                consistency['inconsistencies'].append('Heading out of valid range')
                consistency['consistency_score'] -= 0.1
        
        consistency['consistency_score'] = max(0.0, consistency['consistency_score'])
        
        return consistency
    
    def check_timestamp_validity(self, record: pd.Series) -> Dict[str, Any]:
        timestamp_check = {
            'is_valid': True,
            'issues': []
        }
        
        timestamp = record.get('timestamp')
        if pd.notna(timestamp):
            try:
                parsed_time = pd.to_datetime(timestamp)
                
                if parsed_time > datetime.now():
                    timestamp_check['issues'].append('Timestamp is in the future')
                    timestamp_check['is_valid'] = False
                
                if parsed_time.year < 2000:
                    timestamp_check['issues'].append('Timestamp seems too old')
                    timestamp_check['is_valid'] = False
                    
            except Exception as e:
                timestamp_check['issues'].append(f'Invalid timestamp format: {str(e)}')
                timestamp_check['is_valid'] = False
        else:
            timestamp_check['issues'].append('Timestamp is missing')
            timestamp_check['is_valid'] = False
        
        return timestamp_check
    
    def check_image_existence(self, record: pd.Series) -> Dict[str, Any]:
        image_check = {
            'image_exists': False,
            'image_path': None,
            'issues': []
        }
        
        image_path = record.get('image_path')
        if pd.notna(image_path):
            path = Path(image_path)
            if path.exists():
                image_check['image_exists'] = True
                image_check['image_path'] = str(image_path)
            else:
                image_check['issues'].append(f'Image file not found: {image_path}')
        else:
            image_check['issues'].append('Image path is missing')
        
        return image_check
    
    def perform_audit(self, record: pd.Series, audit_id: int) -> Dict[str, Any]:
        audit_entry = {
            'audit_id': audit_id,
            'timestamp': datetime.now().isoformat(),
            'record_index': record.name,
            'flight_id': record.get('flight_id', 'unknown'),
            'completeness_check': self.check_field_completeness(record),
            'consistency_check': self.check_value_consistency(record),
            'timestamp_check': self.check_timestamp_validity(record),
            'image_check': self.check_image_existence(record),
            'overall_status': 'passed',
            'issues': [],
            'recommendations': []
        }
        
        if audit_entry['completeness_check']['missing_fields']:
            audit_entry['overall_status'] = 'failed'
            audit_entry['issues'].append(f"Missing fields: {', '.join(audit_entry['completeness_check']['missing_fields'])}")
            audit_entry['recommendations'].append('Complete all required metadata fields')
        
        if audit_entry['consistency_check']['inconsistencies']:
            audit_entry['overall_status'] = 'warning'
            audit_entry['issues'].extend(audit_entry['consistency_check']['inconsistencies'])
            audit_entry['recommendations'].append('Review and correct inconsistent values')
        
        if not audit_entry['timestamp_check']['is_valid']:
            audit_entry['overall_status'] = 'failed'
            audit_entry['issues'].extend(audit_entry['timestamp_check']['issues'])
            audit_entry['recommendations'].append('Ensure timestamps are valid and properly formatted')
        
        if not audit_entry['image_check']['image_exists']:
            audit_entry['overall_status'] = 'warning'
            audit_entry['issues'].extend(audit_entry['image_check']['issues'])
            audit_entry['recommendations'].append('Verify image file paths and ensure files exist')
        
        return audit_entry
    
    def generate_audit_log(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting QA audit process...")
        
        sampled_df = self.select_audit_samples(df)
        
        audit_log = {
            'audit_metadata': {
                'audit_timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'sampled_records': len(sampled_df),
                'sample_percentage': self.sample_percentage,
                'auditor': 'automated_qa_system'
            },
            'audit_results': {
                'passed': 0,
                'warning': 0,
                'failed': 0,
                'entries': []
            }
        }
        
        for idx, (_, record) in enumerate(sampled_df.iterrows(), 1):
            audit_entry = self.perform_audit(record, idx)
            audit_log['audit_results']['entries'].append(audit_entry)
            
            status = audit_entry['overall_status']
            audit_log['audit_results'][status] += 1
            
            logger.info(f"Audit {idx}/{len(sampled_df)}: {status.upper()} - Flight ID: {audit_entry['flight_id']}")
        
        audit_log['summary'] = self.generate_summary(audit_log)
        
        return audit_log
    
    def generate_summary(self, audit_log: Dict) -> Dict[str, Any]:
        total = audit_log['audit_metadata']['sampled_records']
        passed = audit_log['audit_results']['passed']
        warning = audit_log['audit_results']['warning']
        failed = audit_log['audit_results']['failed']
        
        summary = {
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'warning_rate': (warning / total * 100) if total > 0 else 0,
            'failure_rate': (failed / total * 100) if total > 0 else 0,
            'overall_quality': 'good' if passed / total >= 0.8 else 'fair' if passed / total >= 0.6 else 'poor',
            'common_issues': self.identify_common_issues(audit_log['audit_results']['entries']),
            'recommendations': self.generate_recommendations(audit_log['audit_results']['entries'])
        }
        
        return summary
    
    def identify_common_issues(self, audit_entries: List[Dict]) -> Dict[str, int]:
        issue_counts = {}
        
        for entry in audit_entries:
            for issue in entry['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
    
    def generate_recommendations(self, audit_entries: List[Dict]) -> List[str]:
        recommendation_counts = {}
        
        for entry in audit_entries:
            for rec in entry['recommendations']:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [rec for rec, count in top_recommendations[:5]]
    
    def save_audit_log(self, audit_log: Dict, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_path = output_path / 'qa_audit_log.json'
        with open(json_path, 'w') as f:
            json.dump(audit_log, f, indent=2, default=str)
        
        logger.info(f"Audit log saved to {json_path}")
        
        csv_path = output_path / 'qa_audit_summary.csv'
        self.save_csv_summary(audit_log, csv_path)
        
        report_path = output_path / 'qa_audit_report.txt'
        self.generate_text_report(audit_log, report_path)
    
    def save_csv_summary(self, audit_log: Dict, csv_path: str):
        summary_data = []
        
        for entry in audit_log['audit_results']['entries']:
            summary_data.append({
                'audit_id': entry['audit_id'],
                'flight_id': entry['flight_id'],
                'status': entry['overall_status'],
                'completeness_score': entry['completeness_check']['completeness_score'],
                'consistency_score': entry['consistency_check']['consistency_score'],
                'timestamp_valid': entry['timestamp_check']['is_valid'],
                'image_exists': entry['image_check']['image_exists'],
                'issues': '; '.join(entry['issues']),
                'recommendations': '; '.join(entry['recommendations'])
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV summary saved to {csv_path}")
    
    def generate_text_report(self, audit_log: Dict, report_path: str):
        report_lines = [
            "Quality Assurance Audit Report",
            "=" * 50,
            f"Audit Timestamp: {audit_log['audit_metadata']['audit_timestamp']}",
            f"Total Records: {audit_log['audit_metadata']['total_records']}",
            f"Sampled Records: {audit_log['audit_metadata']['sampled_records']}",
            f"Sample Percentage: {audit_log['audit_metadata']['sample_percentage']}%",
            "",
            "Audit Results:",
            f"  Passed: {audit_log['audit_results']['passed']} ({audit_log['summary']['pass_rate']:.1f}%)",
            f"  Warning: {audit_log['audit_results']['warning']} ({audit_log['summary']['warning_rate']:.1f}%)",
            f"  Failed: {audit_log['audit_results']['failed']} ({audit_log['summary']['failure_rate']:.1f}%)",
            "",
            f"Overall Quality: {audit_log['summary']['overall_quality'].upper()}",
            "",
            "Common Issues:"
        ]
        
        for issue, count in audit_log['summary']['common_issues'].items():
            report_lines.append(f"  - {issue} ({count} occurrences)")
        
        report_lines.extend([
            "",
            "Recommendations:"
        ])
        
        for rec in audit_log['summary']['recommendations']:
            report_lines.append(f"  - {rec}")
        
        report_content = '\n'.join(report_lines)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Text report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate QA audit log for metadata')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata Excel file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for audit logs')
    parser.add_argument('--sample', type=float, default=5.0, help='Sample percentage for audit (default: 5%)')
    args = parser.parse_args()
    
    auditor = QAAuditLogger(sample_percentage=args.sample)
    df = auditor.load_metadata(args.metadata)
    audit_log = auditor.generate_audit_log(df)
    auditor.save_audit_log(audit_log, args.output)
    
    logger.info(f"QA audit complete! Overall quality: {audit_log['summary']['overall_quality']}")


if __name__ == '__main__':
    main()