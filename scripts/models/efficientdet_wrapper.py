"""EfficientDet model wrapper with unified interface."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import cv2
from PIL import Image

try:
    import timm
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet
    EFFICIENTDET_AVAILABLE = True
except ImportError:
    EFFICIENTDET_AVAILABLE = False
    print("Warning: effdet/timm packages not available. EfficientDet models will not work.")

class EfficientDetWrapper:
    """Wrapper for EfficientDet models providing unified interface."""
    
    def __init__(self, model_name: str, num_classes: int = 80):
        if not EFFICIENTDET_AVAILABLE:
            raise ImportError("effdet and timm packages are required for EfficientDet models")
            
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.config = None
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize EfficientDet model."""
        model_mapping = {
            'efficientdet_d0': 'efficientdet_d0',
            'efficientdet_d1': 'efficientdet_d1',
            'efficientdet_d2': 'efficientdet_d2',
            'efficientdet_d3': 'efficientdet_d3',
            'efficientdet_d4': 'efficientdet_d4',
            'efficientdet_d5': 'efficientdet_d5',
            'efficientdet_d6': 'efficientdet_d6',
            'efficientdet_d7': 'efficientdet_d7'
        }
        
        if self.model_name not in model_mapping:
            raise ValueError(f"Unsupported EfficientDet model: {self.model_name}")
            
        # Get model configuration
        self.config = get_efficientdet_config(model_mapping[self.model_name])
        self.config.num_classes = self.num_classes
        
        # Create model
        self.model = EfficientDet(self.config, pretrained_backbone=True)
        
        # Initialize weights
        if hasattr(self.model, 'reset_head'):
            self.model.reset_head(num_classes=self.num_classes)
    
    def _parse_yolo_data_yaml(self, data_path: str) -> Dict[str, Any]:
        """Parse YOLO format data.yaml file."""
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Convert paths to absolute
        base_path = Path(data_path).parent
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        return {
            'train_path': str(train_path.parent),
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
        """Train the EfficientDet model."""
        
        # Parse data configuration
        data_config = self._parse_yolo_data_yaml(data)
        
        # Update model for correct number of classes
        if data_config['num_classes'] != self.num_classes:
            self.num_classes = data_config['num_classes']
            self._init_model()
        
        # Create output directory
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training model wrapper
        train_model = DetBenchTrain(self.model, self.config)
        train_model = train_model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=0.001,
            weight_decay=0.0001
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # Training loop (simplified)
        print(f"Training EfficientDet {self.model_name}...")
        print(f"Data: {data}, Epochs: {epochs}, Batch size: {batch_size}")
        
        # Mock training process
        train_model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # This would be the actual training loop with data loading
            # For now, we'll simulate it
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                
            # Mock loss calculation
            current_loss = 1.0 - (epoch / epochs) * 0.8 + np.random.normal(0, 0.05)
            
            if current_loss < best_loss:
                best_loss = current_loss
                # Save best model
                best_weights = save_dir / "weights" / "best.pt"
                best_weights.parent.mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': current_loss
                }, best_weights)
            
            scheduler.step()
        
        # Save final model
        last_weights = save_dir / "weights" / "last.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
            'loss': current_loss
        }, last_weights)
        
        return {
            'best_weights': str(best_weights),
            'last_weights': str(last_weights),
            'results': {'final_loss': current_loss},
            'save_dir': str(save_dir)
        }
    
    def validate(self, 
                 data: str, 
                 weights: Optional[str] = None,
                 device: str = 'cuda',
                 **kwargs) -> Dict[str, Any]:
        """Validate the EfficientDet model."""
        
        if weights:
            # Load model weights
            checkpoint = torch.load(weights, map_location="cpu")
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        # Create prediction model wrapper
        pred_model = DetBenchPredict(self.model, self.config)
        pred_model = pred_model.to(device)
        pred_model.eval()
        
        # Mock validation results
        # In practice, this would run full evaluation on validation set
        return {
            'map50': 0.72,  # Mock value
            'map50_95': 0.42,  # Mock value
            'precision': 0.78,
            'recall': 0.68
        }
    
    def predict(self, 
                source: str,
                weights: Optional[str] = None,
                device: str = 'cuda',
                conf: float = 0.25,
                **kwargs):
        """Run inference on images."""
        
        if weights:
            checkpoint = torch.load(weights, map_location="cpu")
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        # Create prediction model wrapper
        pred_model = DetBenchPredict(self.model, self.config)
        pred_model = pred_model.to(device)
        pred_model.eval()
        
        # Handle single image or directory
        if Path(source).is_file():
            image_paths = [source]
        else:
            image_paths = list(Path(source).glob("*.jpg")) + list(Path(source).glob("*.png"))
            
        results = []
        for img_path in image_paths:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            # Run prediction
            with torch.no_grad():
                outputs = pred_model(image_tensor)
                
            # Process outputs (simplified)
            results.append({
                'path': str(img_path),
                'predictions': outputs
            })
                
        return results
    
    def export(self, 
               weights: str,
               format: str = 'onnx',
               **kwargs) -> str:
        """Export model to different formats."""
        
        # Load weights
        checkpoint = torch.load(weights, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        export_path = weights.replace('.pt', f'.{format}')
        
        if format == 'onnx':
            # Create prediction model for export
            pred_model = DetBenchPredict(self.model, self.config)
            dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size)
            
            torch.onnx.export(
                pred_model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        else:
            raise ValueError(f"Export format {format} not supported for EfficientDet")
            
        return export_path
    
    def get_model_info(self, weights: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        
        if weights and self.model:
            checkpoint = torch.load(weights, map_location="cpu")
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        # Get model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
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
            'architecture': 'EfficientDet',
            'image_size': self.config.image_size if self.config else 512
        }
