import argparse
import yaml
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import json
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RTDETRInference:
    def __init__(self, weights_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.classes = self.config.get('classes', {})
        
        self.load_model(weights_path)
    
    def load_model(self, weights_path: str):
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            
            logger.info(f"Loading RT-DETR model from {weights_path}")
            self.model = RTDetrForObjectDetection.from_pretrained(weights_path)
            self.processor = RTDetrImageProcessor.from_pretrained(weights_path)
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.warning(f"Could not load RT-DETR from transformers: {e}")
            logger.info("Loading as PyTorch state dict")
            
            state_dict = torch.load(weights_path, map_location=self.device)
            
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model = self._create_simple_model(len(self.classes))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = None
    
    def _create_simple_model(self, num_classes):
        import torch.nn as nn
        
        class SimpleDetector(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                )
                
                # 目标检测头 - 输出分类和边界框
                self.classification_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
                # 边界框回归头
                self.bbox_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 4)  # 4个边界框坐标
                )
            
            def forward(self, x):
                features = self.backbone(x)
                class_logits = self.classification_head(features)
                bbox_preds = self.bbox_head(features)
                
                # 模拟RT-DETR的输出格式
                class_logits = class_logits.unsqueeze(1)  # [batch_size, 1, num_classes]
                bbox_preds = bbox_preds.unsqueeze(1)     # [batch_size, 1, 4]
                
                # 创建与RT-DETR兼容的输出结构
                class Output:
                    def __init__(self, logits, pred_boxes):
                        self.logits = logits
                        self.pred_boxes = pred_boxes
                
                return Output(logits=class_logits, pred_boxes=bbox_preds)
        
        return SimpleDetector(num_classes)
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        
        if self.processor:
            processed = self.processor(images=image, return_tensors="pt")
            return processed.pixel_values.squeeze(0)
        else:
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            return image_tensor
    
    def postprocess_predictions(self, outputs, image_shape: Tuple[int, int]) -> List[Dict]:
        detections = []
        
        if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
            logits = outputs.logits[0]
            boxes = outputs.pred_boxes[0]
            
            probs = torch.softmax(logits, dim=-1)
            max_probs, class_ids = torch.max(probs, dim=-1)
            
            mask = max_probs > self.confidence_threshold
            
            for i in range(len(mask)):
                if mask[i]:
                    x1, y1, x2, y2 = boxes[i].cpu().numpy()
                    class_id = class_ids[i].item()
                    confidence = max_probs[i].item()
                    
                    x1 = max(0, min(x1, image_shape[1]))
                    y1 = max(0, min(y1, image_shape[0]))
                    x2 = max(0, min(x2, image_shape[1]))
                    y2 = max(0, min(y2, image_shape[0]))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': class_id,
                            'confidence': confidence,
                            'class_name': self.classes.get(str(class_id), f'class_{class_id}')
                        })
        else:
            probs = torch.softmax(outputs, dim=-1)
            max_probs, class_ids = torch.max(probs, dim=-1)
            
            if max_probs > self.confidence_threshold:
                detections.append({
                    'class_id': class_ids.item(),
                    'confidence': max_probs.item(),
                    'class_name': self.classes.get(str(class_ids.item()), f'class_{class_ids.item()}')
                })
        
        return detections
    
    def nms(self, detections: List[Dict]) -> List[Dict]:
        if len(detections) == 0:
            return detections
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def predict_single_image(self, image_path: str) -> List[Dict]:
        image = Image.open(image_path)
        image_shape = image.size[::-1]
        
        image_tensor = self.preprocess_image(image_path).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        detections = self.postprocess_predictions(outputs, image_shape)
        detections = self.nms(detections)
        
        return detections
    
    def save_yolo_format(self, detections: List[Dict], image_shape: Tuple[int, int], 
                        output_path: str):
        height, width = image_shape
        
        with open(output_path, 'w') as f:
            for det in detections:
                if 'bbox' in det:
                    x1, y1, x2, y2 = det['bbox']
                    
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height
                    
                    class_id = det['class_id']
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    
    def save_coco_format(self, all_detections: List[Dict], output_path: str):
        coco_output = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        annotation_id = 0
        for class_id, class_name in self.classes.items():
            coco_output['categories'].append({
                'id': int(class_id),
                'name': class_name
            })
        
        for image_data in all_detections:
            image_id = image_data['image_id']
            image_path = image_data['image_path']
            image_shape = image_data['image_shape']
            
            coco_output['images'].append({
                'id': image_id,
                'file_name': Path(image_path).name,
                'width': image_shape[1],
                'height': image_shape[0]
            })
            
            for detection in image_data['detections']:
                if 'bbox' in detection:
                    x1, y1, x2, y2 = detection['bbox']
                    
                    coco_output['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': detection['class_id'],
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': (x2 - x1) * (y2 - y1),
                        'score': detection['confidence'],
                        'iscrowd': 0
                    })
                    annotation_id += 1
        
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
    
    def process_directory(self, input_dir: str, output_dir: str, format: str = 'yolo'):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        logger.info(f"Found {len(image_files)} images to process")
        
        all_detections = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                detections = self.predict_single_image(str(image_file))
                
                image = Image.open(image_file)
                image_shape = image.size[::-1]
                
                if format == 'yolo':
                    output_file = output_path / f"{image_file.stem}.txt"
                    self.save_yolo_format(detections, image_shape, str(output_file))
                elif format == 'coco':
                    all_detections.append({
                        'image_id': len(all_detections),
                        'image_path': str(image_file),
                        'image_shape': image_shape,
                        'detections': detections
                    })
                
                logger.info(f"Processed {image_file.name}: {len(detections)} detections")
                
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
        
        if format == 'coco':
            output_file = output_path / 'detections_coco.json'
            self.save_coco_format(all_detections, str(output_file))
        
        logger.info(f"Inference complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run RT-DETR inference on images')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for annotations')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--format', type=str, default='yolo', choices=['yolo', 'coco'], 
                       help='Output annotation format')
    args = parser.parse_args()
    
    inference = RTDETRInference(args.weights, args.config)
    inference.process_directory(args.input, args.output, args.format)


if __name__ == '__main__':
    main()