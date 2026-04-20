import argparse
import yaml
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLODataset(Dataset):
    def __init__(self, data_yaml: str, split: str = 'train', transform=None):
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.image_dir = Path(data_config[split])
        self.label_dir = Path(data_config[f'{split}_labels'])
        self.classes = data_config['names']
        self.num_classes = len(self.classes)
        self.transform = transform
        
        self.image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
        logger.info(f"Loaded {len(self.image_files)} images for {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        label_path = self.label_dir / f"{image_path.stem}.txt"
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        x1 = (x_center - width / 2)
                        y1 = (y_center - height / 2)
                        x2 = (x_center + width / 2)
                        y2 = (y_center + height / 2)
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class RTDETRTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.setup_model()
        self.setup_data()
        self.setup_training()
    
    def setup_model(self):
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            
            model_name = self.config.get('model', 'PaddleDetection/RT-DETR/rtdetr_r50vd_6x_coco')
            
            logger.info(f"Loading RT-DETR model: {model_name}")
            self.model = RTDetrForObjectDetection.from_pretrained(
                model_name,
                num_labels=self.config['num_classes'],
                ignore_mismatched_sizes=True
            )
            self.processor = RTDetrImageProcessor.from_pretrained(model_name)
            
        except ImportError:
            logger.warning("Transformers RT-DETR not available, using fallback implementation")
            self.model = self._create_fallback_model()
            self.processor = None
    
    def _create_fallback_model(self):
        class SimpleRTDETR(nn.Module):
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
                    def __init__(self, logits, pred_boxes, loss=None):
                        self.logits = logits
                        self.pred_boxes = pred_boxes
                        self.loss = loss
                
                return Output(logits=class_logits, pred_boxes=bbox_preds)
    
    def setup_data(self):
        data_yaml = self.config['data']
        
        self.train_dataset = YOLODataset(data_yaml, split='train')
        self.val_dataset = YOLODataset(data_yaml, split='val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        images = []
        targets = []
        
        for image, target in batch:
            if self.processor:
                processed = self.processor(images=image, return_tensors="pt")
                images.append(processed.pixel_values.squeeze(0))
            else:
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                images.append(image_tensor)
            targets.append(target)
        
        images = torch.stack(images)
        return images, targets
    
    def setup_training(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.0001)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.epochs = self.config['epochs']
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self._compute_loss(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _compute_loss(self, outputs, targets):
        if hasattr(outputs, 'loss'):
            return outputs.loss
        
        loss = 0
        classification_loss = 0
        bbox_loss = 0
        
        for i, target in enumerate(targets):
            if len(target['labels']) > 0:
                labels = target['labels'].to(self.device)
                boxes = target['boxes'].to(self.device)
                
                # 提取分类预测
                if hasattr(outputs, 'logits'):
                    # 新的回退模型输出格式
                    pred_logits = outputs.logits[i].squeeze(0)
                    pred_boxes = outputs.pred_boxes[i].squeeze(0)
                else:
                    # 旧的输出格式（如果有的话）
                    pred_logits = outputs[i]
                    pred_boxes = None
                
                # 分类损失
                classification_loss += self.criterion(pred_logits, labels)
                
                # 边界框回归损失
                if pred_boxes is not None and len(boxes) > 0:
                    # 使用L1损失作为边界框回归损失
                    bbox_loss += nn.functional.l1_loss(pred_boxes, boxes[0])
        
        total_loss = classification_loss + bbox_loss
        return total_loss / len(targets) if total_loss > 0 else torch.tensor(0.0, device=self.device)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info("Training complete!")
        self.save_final_model()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pt'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def save_final_model(self):
        output_path = self.output_dir / 'rtdetr_neodrone.pt'
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"Final model saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train RT-DETR model for NEODrone dataset')
    parser.add_argument('--config', type=str, required=True, help='Training configuration file')
    args = parser.parse_args()
    
    trainer = RTDETRTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()