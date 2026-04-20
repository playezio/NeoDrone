#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DatasetCardGenerator:
    def __init__(self, dataset_name: str = "NEODrone", version: str = "2.1"):
        self.dataset_name = dataset_name
        self.version = version
        self.base_info = {
            "dataset_name": dataset_name,
            "version": version,
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "license": "CC BY 4.0",
            "homepage": "https://github.com/your-org/NEODrone",
            "repository": "https://github.com/your-org/NEODrone",
            "paper": "https://arxiv.org/abs/xxxx.xxxxx"
        }
    
    def generate_huggingface_card(self, metadata: Dict[str, Any]) -> str:
        card_lines = [
            "---",
            f"license: {self.base_info['license']}",
            f"homepage: {self.base_info['homepage']}",
            f"repository: {self.base_info['repository']}",
            f"paper: {self.base_info['paper']}",
            "task_categories:",
            "  - object-detection",
            "  - image-classification",
            "language:",
            "  - en",
            "size_categories:",
            "  - 100K<n<1M",
            "---",
            "",
            f"# {self.dataset_name} Dataset",
            "",
            f"## Dataset Description",
            "",
            f"{self.dataset_name} is a comprehensive drone imagery dataset designed for computer vision research and applications. It contains multi-modal imagery (visible light and thermal) captured from various altitudes and perspectives across diverse scenes.",
            "",
            f"## Dataset Version",
            "",
            f"- **Version**: {self.version}",
            f"- **Release Date**: {self.base_info['creation_date']}",
            "",
            f"## Dataset Statistics",
            "",
            f"- **Total Images**: {metadata.get('total_images', 'N/A')}",
            f"- **Visible Light Images**: {metadata.get('visible_images', 'N/A')}",
            f"- **Thermal Images**: {metadata.get('thermal_images', 'N/A')}",
            f"- **Image Pairs**: {metadata.get('image_pairs', 'N/A')}",
            f"- **Annotations**: {metadata.get('total_annotations', 'N/A')}",
            f"- **Object Categories**: {metadata.get('num_classes', 28)}",
            f"- **Resolution**: {metadata.get('resolution', '1920x1080')}",
            "",
            f"## Data Collection",
            "",
            f"The dataset was collected using DJI drones equipped with high-resolution visible and thermal cameras. Data was captured across multiple locations and weather conditions to ensure diversity and robustness.",
            "",
            f"### Acquisition Parameters",
            "",
            f"- **Altitude Range**: {metadata.get('altitude_range', '10-500m')}",
            f"- **Gimbal Pitch Range**: {metadata.get('gimbal_pitch_range', '-90° to 0°')}",
            f"- **Weather Conditions**: {metadata.get('weather_conditions', 'sunny, cloudy, rainy')}",
            f"- **Scene Types**: {metadata.get('scene_types', 'urban, suburban, rural, industrial')}",
            "",
            f"## Dataset Structure",
            "",
            "```",
            "dataset/",
            "├── visible/",
            "│   ├── train/",
            "│   ├── val/",
            "│   └── test/",
            "├── thermal/",
            "│   ├── train/",
            "│   ├── val/",
            "│   └── test/",
            "├── annotations/",
            "│   ├── train/",
            "│   ├── val/",
            "│   └── test/",
            "└── metadata/",
            "    └── metadata.xlsx",
            "```",
            "",
            f"## Object Categories",
            "",
            f"The dataset includes {metadata.get('num_classes', 28)} object categories:",
            ""
        ]
        
        categories = metadata.get('categories', [
            "person", "car", "truck", "bus", "motorcycle", "bicycle", "boat", "airplane", "train",
            "building", "road", "bridge", "traffic_light", "traffic_sign", "tree", "pole", "fence",
            "power_line", "tower", "container", "crane", "excavator", "bulldozer", "tractor", "drone",
            "helicopter", "animal", "other"
        ])
        
        for i, category in enumerate(categories, 1):
            card_lines.append(f"{i}. {category}")
        
        card_lines.extend([
            "",
            f"## Annotation Format",
            "",
            "Annotations are provided in YOLO format with the following structure:",
            "",
            "```\n",
            "<class_id> <x_center> <y_center> <width> <height>\n",
            "```\n",
            "",
            "Where:",
            "- `<class_id>`: Integer class identifier (0-{})".format(len(categories) - 1),
            "- `<x_center>`, `<y_center>`: Normalized center coordinates (0-1)",
            "- `<width>`, `<height>`: Normalized bounding box dimensions (0-1)",
            "",
            f"## Metadata",
            "",
            "Each image is accompanied by rich metadata including:",
            "- GPS coordinates (latitude, longitude, altitude)",
            "- Drone attitude (heading, pitch, roll)",
            "- Gimbal orientation (pitch, roll)",
            "- Camera settings (zoom, focus)",
            "- Acquisition conditions (weather, scene type)",
            "- Sensor type (visible/thermal)",
            "",
            f"## Data Splits",
            "",
            f"- **Training Set**: {metadata.get('train_split', '70%')}",
            f"- **Validation Set**: {metadata.get('val_split', '15%')}",
            f"- **Test Set**: {metadata.get('test_split', '15%')}",
            "",
            f"## Data Quality",
            "",
            f"- **Frame Alignment**: ≤5 pixels between visible and thermal images",
            f"- **Time Synchronization**: ≤10ms between modalities",
            f"- **Annotation Quality**: Krippendorff's α ≥ 0.8",
            f"- **Metadata Completeness**: ≥95% complete records",
            "",
            f"## Use Cases",
            "",
            "- Object detection in aerial imagery",
            "- Multi-modal image analysis",
            "- Drone-based surveillance and monitoring",
            "- Search and rescue operations",
            "- Infrastructure inspection",
            "- Agricultural monitoring",
            "- Traffic management",
            "",
            f"## Citation",
            "",
            "If you use this dataset in your research, please cite:",
            "",
            "```bibtex",
            "@dataset{neodrone2024,",
            "  title={NEODrone: A Multi-Modal Drone Imagery Dataset for Computer Vision},",
            "  author={Your Name and Others},",
            "  year={2024},",
            "  publisher={GitHub},",
            "  version={},".format(self.version),
            "  url={}".format(self.base_info['repository']),
            "}",
            "```",
            "",
            f"## License",
            "",
            f"This dataset is licensed under the {self.base_info['license']} license.",
            "",
            f"## Acknowledgments",
            "",
            "We thank all contributors to the NEODrone project and the drone operators who helped collect this data.",
            "",
            f"## Contact",
            "",
            f"For questions or feedback, please contact us at: support@neodrone.org"
        ])
        
        return '\n'.join(card_lines)
    
    def generate_science_db_card(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        card = {
            "title": f"{self.dataset_name} Dataset",
            "description": f"A comprehensive multi-modal drone imagery dataset for computer vision research",
            "creators": [
                "Your Name",
                "Other Contributors"
            ],
            "publication_date": self.base_info['creation_date'],
            "version": self.version,
            "license": self.base_info['license'],
            "keywords": [
                "drone imagery",
                "object detection",
                "multi-modal",
                "computer vision",
                "aerial photography",
                "thermal imaging",
                "visible light",
                "remote sensing"
            ],
            "subjects": [
                "Computer Science",
                "Artificial Intelligence",
                "Remote Sensing",
                "Image Processing"
            ],
            "resource_type": "dataset",
            "size": metadata.get('total_size', 'Unknown'),
            "formats": [
                "JPG",
                "PNG",
                "TXT",
                "XLSX"
            ],
            "language": "en",
            "related_identifiers": [
                {
                    "relation": "isSupplementTo",
                    "identifier": self.base_info['paper'],
                    "resource_type": "publication-article"
                },
                {
                    "relation": "isSupplementedBy",
                    "identifier": self.base_info['repository'],
                    "resource_type": "software"
                }
            ],
            "dates": [
                {
                    "type": "created",
                    "date": self.base_info['creation_date']
                },
                {
                    "type": "published",
                    "date": self.base_info['creation_date']
                }
            ],
            "funding": [
                {
                    "name": "Research Grant Name",
                    "identifier": "Grant ID"
                }
            ],
            "access_right": "open",
            "embargo_date": None,
            "access_conditions": None,
            "metadata": {
                "total_images": metadata.get('total_images', 'N/A'),
                "visible_images": metadata.get('visible_images', 'N/A'),
                "thermal_images": metadata.get('thermal_images', 'N/A'),
                "image_pairs": metadata.get('image_pairs', 'N/A'),
                "total_annotations": metadata.get('total_annotations', 'N/A'),
                "num_classes": metadata.get('num_classes', 28),
                "resolution": metadata.get('resolution', '1920x1080'),
                "altitude_range": metadata.get('altitude_range', '10-500m'),
                "gimbal_pitch_range": metadata.get('gimbal_pitch_range', '-90° to 0°'),
                "weather_conditions": metadata.get('weather_conditions', 'sunny, cloudy, rainy'),
                "scene_types": metadata.get('scene_types', 'urban, suburban, rural, industrial'),
                "train_split": metadata.get('train_split', '70%'),
                "val_split": metadata.get('val_split', '15%'),
                "test_split": metadata.get('test_split', '15%')
            }
        }
        
        return card
    
    def generate_kaggle_card(self, metadata: Dict[str, Any]) -> str:
        card_lines = [
            f"# {self.dataset_name} Dataset",
            "",
            f"## Overview",
            "",
            f"{self.dataset_name} is a comprehensive drone imagery dataset designed for computer vision research and applications. It contains multi-modal imagery (visible light and thermal) captured from various altitudes and perspectives across diverse scenes.",
            "",
            f"## Dataset Contents",
            "",
            f"- **Total Images**: {metadata.get('total_images', 'N/A')}",
            f"- **Visible Light Images**: {metadata.get('visible_images', 'N/A')}",
            f"- **Thermal Images**: {metadata.get('thermal_images', 'N/A')}",
            f"- **Image Pairs**: {metadata.get('image_pairs', 'N/A')}",
            f"- **Annotations**: {metadata.get('total_annotations', 'N/A')}",
            f"- **Object Categories**: {metadata.get('num_classes', 28)}",
            "",
            f"## Files",
            "",
            "The dataset is organized as follows:",
            "",
            "- `visible/`: Visible light images",
            "- `thermal/`: Thermal images",
            "- `annotations/`: YOLO format annotations",
            "- `metadata/`: Rich metadata for each image",
            "",
            f"## Object Categories",
            "",
            "The dataset includes the following object categories:",
            ""
        ]
        
        categories = metadata.get('categories', [
            "person", "car", "truck", "bus", "motorcycle", "bicycle", "boat", "airplane", "train",
            "building", "road", "bridge", "traffic_light", "traffic_sign", "tree", "pole", "fence",
            "power_line", "tower", "container", "crane", "excavator", "bulldozer", "tractor", "drone",
            "helicopter", "animal", "other"
        ])
        
        for i, category in enumerate(categories, 1):
            card_lines.append(f"{i}. {category}")
        
        card_lines.extend([
            "",
            f"## Citation",
            "",
            "If you use this dataset, please cite:",
            "",
            "```bibtex",
            "@dataset{neodrone2024,",
            "  title={NEODrone: A Multi-Modal Drone Imagery Dataset for Computer Vision},",
            "  author={Your Name and Others},",
            "  year={2024},",
            "  version={},".format(self.version),
            "}",
            "```",
            "",
            f"## License",
            "",
            f"This dataset is licensed under {self.base_info['license']}."
        ])
        
        return '\n'.join(card_lines)
    
    def save_cards(self, metadata: Dict[str, Any], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hf_card = self.generate_huggingface_card(metadata)
        hf_card_path = output_path / "README.md"
        with open(hf_card_path, 'w', encoding='utf-8') as f:
            f.write(hf_card)
        print(f"Hugging Face card saved to: {hf_card_path}")
        
        science_db_card = self.generate_science_db_card(metadata)
        science_db_card_path = output_path / "science_db_card.json"
        with open(science_db_card_path, 'w', encoding='utf-8') as f:
            json.dump(science_db_card, f, indent=2, ensure_ascii=False)
        print(f"ScienceDB card saved to: {science_db_card_path}")
        
        kaggle_card = self.generate_kaggle_card(metadata)
        kaggle_card_path = output_path / "kaggle_card.md"
        with open(kaggle_card_path, 'w', encoding='utf-8') as f:
            f.write(kaggle_card)
        print(f"Kaggle card saved to: {kaggle_card_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset cards for NEODrone dataset')
    parser.add_argument('--dataset_name', type=str, default='NEODrone', help='Dataset name')
    parser.add_argument('--version', type=str, default='2.1', help='Dataset version')
    parser.add_argument('--output', type=str, required=True, help='Output directory for dataset cards')
    parser.add_argument('--metadata', type=str, help='Path to metadata JSON file')
    args = parser.parse_args()
    
    generator = DatasetCardGenerator(dataset_name=args.dataset_name, version=args.version)
    
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'total_images': 100000,
            'visible_images': 50000,
            'thermal_images': 50000,
            'image_pairs': 50000,
            'total_annotations': 250000,
            'num_classes': 28,
            'resolution': '1920x1080',
            'altitude_range': '10-500m',
            'gimbal_pitch_range': '-90° to 0°',
            'weather_conditions': 'sunny, cloudy, rainy',
            'scene_types': 'urban, suburban, rural, industrial',
            'train_split': '70%',
            'val_split': '15%',
            'test_split': '15%'
        }
    
    generator.save_cards(metadata, args.output)
    print("Dataset cards generated successfully!")


if __name__ == '__main__':
    main()