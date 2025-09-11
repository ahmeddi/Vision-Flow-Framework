"""YOLO-NAS model wrapper with unified interface."""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
from PIL import Image
import cv2

try:
    from super_gradients.training import models
    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train, 
        coco_detection_yolo_format_val
    )
    from super_gradients.training import Trainer
    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    YOLO_NAS_AVAILABLE = True
except ImportError:
    YOLO_NAS_AVAILABLE = False
    print("Warning: super-gradients not available. YOLO-NAS models will not work.")

class YOLONASWrapper:
    """Wrapper for YOLO-NAS models providing unified interface."""
    
    def __init__(self, model_name: str, num_classes: int = 80):
        if not YOLO_NAS_AVAILABLE:
            raise ImportError("super-gradients package is required for YOLO-NAS models")
            
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.trainer = None
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize YOLO-NAS model."""
        model_mapping = {
            'yolo_nas_s': 'yolo_nas_s',
            'yolo_nas_m': 'yolo_nas_m', 
            'yolo_nas_l': 'yolo_nas_l'
        }
        
        if self.model_name not in model_mapping:
            raise ValueError(f"Unsupported YOLO-NAS model: {self.model_name}")
            
        # Load pre-trained model
        self.model = models.get(
            model_mapping[self.model_name],
            num_classes=self.num_classes,
            pretrained_weights="coco"
        )
    
    def _parse_yolo_data_yaml(self, data_path: str) -> Dict[str, Any]:
        """Parse YOLO format data.yaml file."""
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Convert paths to absolute
        base_path = Path(data_path).parent
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        return {
            'train_path': str(train_path.parent),  # images folder parent
            'val_path': str(val_path.parent),
            'num_classes': data_config['nc'],
            'class_names': data_config['names']
        }
    
    def train(self, 
              data: str, 
              epochs: int = 100,
              batch_size: int = 16,
              device: str = 'cuda',
              project: str = 'runs/train',
              name: str = 'exp',
              **kwargs) -> Dict[str, Any]:
        """Train the YOLO-NAS model."""
        
        # Parse data configuration
        data_config = self._parse_yolo_data_yaml(data)
        
        # Update model for correct number of classes
        if data_config['num_classes'] != self.num_classes:
            self.num_classes = data_config['num_classes']
            self._init_model()
        
        # Setup trainer
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.trainer = Trainer(
            experiment_name=name,
            ckpt_root_dir=str(save_dir)
        )
        
        # Prepare dataloaders
        train_dataloader = coco_detection_yolo_format_train(
            dataset_dir=data_config['train_path'],
            images_dir="images/train",
            labels_dir="labels/train", 
            classes=data_config['class_names'],
            input_dim=(640, 640),
            batch_size=batch_size,
            cache_dir=str(save_dir / "cache"),
            **kwargs
        )
        
        val_dataloader = coco_detection_yolo_format_val(
            dataset_dir=data_config['val_path'],
            images_dir="images/val",
            labels_dir="labels/val",
            classes=data_config['class_names'], 
            input_dim=(640, 640),
            batch_size=batch_size,
            cache_dir=str(save_dir / "cache"),
            **kwargs
        )
        
        # Training parameters
        train_params = {
            "silent_mode": False,
            "average_best_models": True,
            "warmup_mode": "linear_epoch_step",
            "warmup_initial_lr": 1e-6,
            "lr_warmup_epochs": 3,
            "initial_lr": 5e-4,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.1,
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.0001},
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {"decay": 0.9, "decay_type": "threshold"},
            "max_epochs": epochs,
            "mixed_precision": True,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=self.num_classes,
                reg_max=16
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=self.num_classes,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7
                    )
                )
            ],
            "metric_to_watch": 'mAP@0.50'
        }
        
        # Start training
        results = self.trainer.train(
            model=self.model,
            training_params=train_params,
            train_loader=train_dataloader,
            valid_loader=val_dataloader
        )
        
        # Save best model
        best_model_path = save_dir / "weights" / "best.pth"
        best_model_path.parent.mkdir(exist_ok=True)
        self.trainer.save_model(str(best_model_path))
        
        return {
            'best_weights': str(best_model_path),
            'results': results,
            'save_dir': str(save_dir)
        }
    
    def validate(self, 
                 data: str, 
                 weights: Optional[str] = None,
                 device: str = 'cuda',
                 **kwargs) -> Dict[str, Any]:
        """Validate the YOLO-NAS model."""
        
        # Parse data configuration
        data_config = self._parse_yolo_data_yaml(data)
        
        if weights:
            # Load model from weights
            model = models.get(
                self.model_name,
                num_classes=data_config['num_classes'],
                checkpoint_path=weights
            )
        else:
            model = self.model
            
        # Prepare validation dataloader
        val_dataloader = coco_detection_yolo_format_val(
            dataset_dir=data_config['val_path'],
            images_dir="images/val", 
            labels_dir="labels/val",
            classes=data_config['class_names'],
            input_dim=(640, 640),
            batch_size=16
        )
        
        # Run validation
        if self.trainer is None:
            self.trainer = Trainer(experiment_name="validation")
            
        val_results = self.trainer.test(
            model=model,
            test_loader=val_dataloader,
            test_metrics_list=[
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=data_config['num_classes'],
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000, 
                        max_predictions=300,
                        nms_threshold=0.7
                    )
                )
            ]
        )
        
        return {
            'map50': val_results.get('mAP@0.50', 0.0),
            'map50_95': val_results.get('mAP@0.50:0.95', 0.0),
            'precision': val_results.get('Precision', 0.0),
            'recall': val_results.get('Recall', 0.0)
        }
    
    def predict(self, 
                source: str,
                weights: Optional[str] = None,
                device: str = 'cuda',
                conf: float = 0.25,
                **kwargs):
        """Run inference on images."""
        
        if weights:
            model = models.get(
                self.model_name,
                num_classes=self.num_classes,
                checkpoint_path=weights
            )
        else:
            model = self.model
            
        model.eval()
        model = model.to(device)
        
        # Handle single image or directory
        if Path(source).is_file():
            image_paths = [source]
        else:
            image_paths = list(Path(source).glob("*.jpg")) + list(Path(source).glob("*.png"))
            
        results = []
        for img_path in image_paths:
            # Load and preprocess image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run prediction
            predictions = model.predict(image, conf=conf)
            results.append({
                'path': str(img_path),
                'predictions': predictions
            })
            
        return results
    
    def export(self, 
               weights: str,
               format: str = 'onnx',
               **kwargs) -> str:
        """Export model to different formats."""
        
        model = models.get(
            self.model_name,
            num_classes=self.num_classes,
            checkpoint_path=weights
        )
        
        export_path = weights.replace('.pth', f'.{format}')
        
        if format == 'onnx':
            dummy_input = torch.randn(1, 3, 640, 640)
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        else:
            raise ValueError(f"Export format {format} not supported for YOLO-NAS")
            
        return export_path
    
    def get_model_info(self, weights: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        
        if weights:
            model = models.get(
                self.model_name,
                num_classes=self.num_classes,
                checkpoint_path=weights
            )
        else:
            model = self.model
            
        # Get model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model size
        if weights:
            model_size_mb = Path(weights).stat().st_size / (1024 * 1024)
        else:
            model_size_mb = 0
            
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'architecture': 'YOLO-NAS'
        }
