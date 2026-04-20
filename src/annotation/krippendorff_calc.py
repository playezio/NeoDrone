import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnnotationConsistencyCalculator:
    def __init__(self, num_classes: int = 28):
        self.num_classes = num_classes
    
    def parse_yolo_annotation(self, annotation_path: str) -> List[Tuple[int, List[float]]]:
        annotations = []
        
        if not Path(annotation_path).exists():
            return annotations
        
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    annotations.append((class_id, bbox))
        
        return annotations
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def match_annotations(self, annotations1: List[Tuple[int, List[float]]], 
                          annotations2: List[Tuple[int, List[float]]], 
                          iou_threshold: float = 0.5) -> List[Tuple]:
        matches = []
        used2 = set()
        
        for class_id1, bbox1 in annotations1:
            best_match = None
            best_iou = 0.0
            
            for idx2, (class_id2, bbox2) in enumerate(annotations2):
                if idx2 in used2:
                    continue
                
                if class_id1 == class_id2:
                    iou = self.calculate_iou(bbox1, bbox2)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = (idx2, class_id2, bbox2)
            
            if best_match and best_iou >= iou_threshold:
                matches.append((class_id1, bbox1, best_match[1], best_match[2], best_iou))
                used2.add(best_match[0])
        
        return matches
    
    def calculate_pairwise_agreement(self, annotations1: List[Tuple[int, List[float]]], 
                                    annotations2: List[Tuple[int, List[float]]], 
                                    iou_threshold: float = 0.5) -> Dict:
        matches = self.match_annotations(annotations1, annotations2, iou_threshold)
        
        total_objects = max(len(annotations1), len(annotations2))
        
        if total_objects == 0:
            return {
                'agreement': 1.0,
                'matches': 0,
                'total_objects': 0,
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0
            }
        
        agreement = len(matches) / total_objects
        
        precision = len(matches) / len(annotations1) if len(annotations1) > 0 else 0.0
        recall = len(matches) / len(annotations2) if len(annotations2) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'agreement': agreement,
            'matches': len(matches),
            'total_objects': total_objects,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def calculate_krippendorff_alpha(self, reliability_matrix: np.ndarray) -> float:
        n, m = reliability_matrix.shape
        
        if n == 0 or m == 0:
            return 1.0
        
        observed_disagreement = 0.0
        expected_disagreement = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    if reliability_matrix[i, j] != reliability_matrix[i, k]:
                        observed_disagreement += 1.0
        
        observed_disagreement /= (n * m * (m - 1))
        
        for j in range(m):
            for k in range(m):
                if j != k:
                    count_j = np.sum(reliability_matrix == j)
                    count_k = np.sum(reliability_matrix == k)
                    expected_disagreement += count_j * count_k
        
        if n * m * (n * m - 1) == 0:
            return 1.0
        
        expected_disagreement /= (n * m * (n * m - 1))
        
        if expected_disagreement == 0:
            return 1.0
        
        alpha = 1.0 - (observed_disagreement / expected_disagreement)
        return max(-1.0, min(1.0, alpha))
    
    def calculate_nominal_alpha(self, annotations_list: List[List[int]]) -> float:
        if not annotations_list or len(annotations_list[0]) == 0:
            return 1.0
        
        reliability_matrix = np.array(annotations_list)
        return self.calculate_krippendorff_alpha(reliability_matrix)
    
    def process_annotation_directories(self, annotator1_dir: str, annotator2_dir: str, 
                                      output_dir: str) -> Dict:
        annotator1_path = Path(annotator1_dir)
        annotator2_path = Path(annotator2_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        annotator1_files = sorted(list(annotator1_path.glob('*.txt')))
        
        results = {
            'total_images': 0,
            'pairwise_agreements': [],
            'class_agreements': defaultdict(list),
            'overall_agreement': 0.0,
            'overall_precision': 0.0,
            'overall_recall': 0.0,
            'overall_f1': 0.0,
            'class_statistics': {}
        }
        
        for annotator1_file in annotator1_files:
            annotator2_file = annotator2_path / annotator1_file.name
            
            if not annotator2_file.exists():
                logger.warning(f"No matching annotation for {annotator1_file.name}")
                continue
            
            annotations1 = self.parse_yolo_annotation(str(annotator1_file))
            annotations2 = self.parse_yolo_annotation(str(annotator2_file))
            
            agreement = self.calculate_pairwise_agreement(annotations1, annotations2)
            results['pairwise_agreements'].append(agreement)
            results['total_images'] += 1
            
            for class_id, bbox1 in annotations1:
                results['class_agreements'][class_id].append(agreement)
        
        if results['total_images'] > 0:
            results['overall_agreement'] = np.mean([a['agreement'] for a in results['pairwise_agreements']])
            results['overall_precision'] = np.mean([a['precision'] for a in results['pairwise_agreements']])
            results['overall_recall'] = np.mean([a['recall'] for a in results['pairwise_agreements']])
            results['overall_f1'] = np.mean([a['f1'] for a in results['pairwise_agreements']])
        
        for class_id, agreements in results['class_agreements'].items():
            results['class_statistics'][class_id] = {
                'count': len(agreements),
                'mean_agreement': np.mean([a['agreement'] for a in agreements]),
                'mean_precision': np.mean([a['precision'] for a in agreements]),
                'mean_recall': np.mean([a['recall'] for a in agreements]),
                'mean_f1': np.mean([a['f1'] for a in agreements])
            }
        
        output_file = output_path / 'consistency_report.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Consistency report saved to {output_file}")
        
        return results
    
    def calculate_alpha_from_dataframe(self, df: pd.DataFrame, 
                                       annotator_cols: List[str], 
                                       value_col: str = 'value') -> float:
        reliability_data = []
        
        for _, row in df.iterrows():
            values = []
            for col in annotator_cols:
                if pd.notna(row[col]):
                    values.append(int(row[col]))
            
            if values:
                reliability_data.append(values)
        
        if not reliability_data:
            return 1.0
        
        max_length = max(len(row) for row in reliability_data)
        padded_data = [row + [row[-1]] * (max_length - len(row)) for row in reliability_data]
        
        reliability_matrix = np.array(padded_data)
        return self.calculate_krippendorff_alpha(reliability_matrix)
    
    def generate_summary_report(self, results: Dict, output_path: str):
        summary_lines = [
            "Annotation Consistency Summary Report",
            "=" * 50,
            f"Total Images Analyzed: {results['total_images']}",
            "",
            "Overall Metrics:",
            f"  Agreement: {results['overall_agreement']:.4f}",
            f"  Precision: {results['overall_precision']:.4f}",
            f"  Recall: {results['overall_recall']:.4f}",
            f"  F1 Score: {results['overall_f1']:.4f}",
            "",
            "Class-wise Statistics:"
        ]
        
        for class_id, stats in sorted(results['class_statistics'].items()):
            summary_lines.extend([
                f"  Class {class_id}:",
                f"    Count: {stats['count']}",
                f"    Agreement: {stats['mean_agreement']:.4f}",
                f"    Precision: {stats['mean_precision']:.4f}",
                f"    Recall: {stats['mean_recall']:.4f}",
                f"    F1 Score: {stats['mean_f1']:.4f}"
            ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Summary report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate annotation consistency using Krippendorff\'s alpha')
    parser.add_argument('--annotator1', type=str, required=True, help='Directory containing first annotator\'s annotations')
    parser.add_argument('--annotator2', type=str, required=True, help='Directory containing second annotator\'s annotations')
    parser.add_argument('--output', type=str, required=True, help='Output directory for reports')
    parser.add_argument('--num-classes', type=int, default=28, help='Number of classes')
    args = parser.parse_args()
    
    calculator = AnnotationConsistencyCalculator(num_classes=args.num_classes)
    results = calculator.process_annotation_directories(args.annotator1, args.annotator2, args.output)
    
    summary_path = Path(args.output) / 'summary.txt'
    calculator.generate_summary_report(results, str(summary_path))
    
    logger.info(f"Consistency calculation complete!")
    logger.info(f"Overall agreement: {results['overall_agreement']:.4f}")
    logger.info(f"Overall F1 score: {results['overall_f1']:.4f}")


if __name__ == '__main__':
    main()