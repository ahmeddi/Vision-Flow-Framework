"""YOLO model wrapper (v8, v11, v7) with unified interface."""
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, Optional
import json

class YOLOWrapper:
    """Wrapper for YOLO models (v8, v11, v7) providing unified interface."""
    
    def __init__(self, model_name: str, num_classes: Optional[int] = None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = YOLO(model_name)
        
    def train(self, 
              data: str, 
              epochs: int = 100,
              batch_size: int = 16,
              device: str = 'cuda',
              project: str = 'runs/train',
              name: str = 'exp',
              **kwargs) -> Dict[str, Any]:
        """Train the model with given parameters."""
        
        # Prepare training arguments
        train_args = {
            'data': data,
            'epochs': epochs,
            'batch': batch_size,
            'device': device,
            'project': project,
            'name': name,
            'save': True,
            'save_period': -1,  # Save only at the end
            'cache': False,
            'amp': True,  # Automatic Mixed Precision
            'verbose': True,
            **kwargs
        }
        
        # Start training
        results = self.model.train(**train_args)
        
        return {
            'best_weights': str(results.save_dir / 'weights' / 'best.pt'),
            'last_weights': str(results.save_dir / 'weights' / 'last.pt'),
            'results': results,
            'save_dir': str(results.save_dir)
        }
    
    def validate(self, 
                 data: str, 
                 weights: Optional[str] = None,
                 device: str = 'cuda',
                 **kwargs) -> Dict[str, Any]:
        """Validate the model."""
        
        if weights:
            model = YOLO(weights)
        else:
            model = self.model
            
        val_results = model.val(
            data=data,
            device=device,
            **kwargs
        )
        
        return {
            'map50': float(val_results.box.map50),
            'map50_95': float(val_results.box.map),
            'mp': float(val_results.box.mp),
            'mr': float(val_results.box.mr),
            'map_per_class': val_results.box.maps.tolist() if hasattr(val_results.box, 'maps') else [],
            'fitness': float(val_results.fitness) if hasattr(val_results, 'fitness') else None
        }
    
    def predict(self, 
                source: str,
                weights: Optional[str] = None,
                device: str = 'cuda',
                conf: float = 0.25,
                iou: float = 0.45,
                **kwargs):
        """Run inference on images."""
        
        if weights:
            model = YOLO(weights)
        else:
            model = self.model
            
        results = model.predict(
            source=source,
            device=device,
            conf=conf,
            iou=iou,
            **kwargs
        )
        
        return results
    
    def export(self, 
               weights: str,
               format: str = 'onnx',
               **kwargs) -> str:
        """Export model to different formats."""
        
        model = YOLO(weights)
        export_path = model.export(
            format=format,
            **kwargs
        )
        
        return str(export_path)
    
    def get_model_info(self, weights: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        
        if weights:
            model = YOLO(weights)
        else:
            model = self.model
            
        # Get model parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
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
            'architecture': 'YOLO'
        }
