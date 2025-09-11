"""Extended training script with support for multiple model families."""
import argparse
import yaml
import os
import json
import time
from pathlib import Path
from ultralytics import YOLO

# Additional model support (to be implemented)
try:
    from super_gradients.training import models as sg_models
    YOLO_NAS_AVAILABLE = True
except ImportError:
    YOLO_NAS_AVAILABLE = False

try:
    import yolox
    YOLOX_AVAILABLE = True
except ImportError:
    YOLOX_AVAILABLE = False

YOLO_MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt',
    'yolov7.pt',  # To be added
]

YOLO_NAS_MODELS = [
    'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'
] if YOLO_NAS_AVAILABLE else []

YOLOX_MODELS = [
    'yolox_nano', 'yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
] if YOLOX_AVAILABLE else []

ALL_MODELS = YOLO_MODELS + YOLO_NAS_MODELS + YOLOX_MODELS

class ModelFactory:
    """Factory class to create different types of models with unified interface."""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int):
        """Create model instance based on model name."""
        if model_name in YOLO_MODELS:
            return YOLOWrapper(model_name)
        elif model_name in YOLO_NAS_MODELS and YOLO_NAS_AVAILABLE:
            return YOLONASWrapper(model_name, num_classes)
        elif model_name in YOLOX_MODELS and YOLOX_AVAILABLE:
            return YOLOXWrapper(model_name, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

class YOLOWrapper:
    """Wrapper for YOLO models (v8, v11, v7)."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = YOLO(model_name)
    
    def train(self, data: str, **kwargs):
        return self.model.train(data=data, **kwargs)
    
    def validate(self, data: str, **kwargs):
        return self.model.val(data=data, **kwargs)

class YOLONASWrapper:
    """Wrapper for YOLO-NAS models."""
    
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.num_classes = num_classes
        # Implementation needed
        
    def train(self, data: str, **kwargs):
        # Implement YOLO-NAS training
        raise NotImplementedError("YOLO-NAS training not implemented")
    
    def validate(self, data: str, **kwargs):
        # Implement YOLO-NAS validation
        raise NotImplementedError("YOLO-NAS validation not implemented")

class YOLOXWrapper:
    """Wrapper for YOLOX models."""
    
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.num_classes = num_classes
        # Implementation needed
        
    def train(self, data: str, **kwargs):
        # Implement YOLOX training
        raise NotImplementedError("YOLOX training not implemented")
    
    def validate(self, data: str, **kwargs):
        # Implement YOLOX validation
        raise NotImplementedError("YOLOX validation not implemented")

def load_config(path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_num_classes_from_data_yaml(data_path: str) -> int:
    """Extract number of classes from data YAML file."""
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config.get('nc', len(data_config.get('names', [])))

def main():
    parser = argparse.ArgumentParser(description='Unified training script for multiple model architectures')
    parser.add_argument('--config', default='configs/base.yaml', help='Training configuration file')
    parser.add_argument('--data', required=True, help='Path to YOLO data YAML defining train/val sets')
    parser.add_argument('--models', nargs='*', default=['yolov8n.pt'], 
                       choices=ALL_MODELS, help='Models to train')
    parser.add_argument('--output', default='results/runs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    import torch
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load configuration
    cfg = load_config(args.config)
    num_classes = get_num_classes_from_data_yaml(args.data)
    
    # Create output directory
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    
    results_summary = []
    
    for model_name in args.models:
        print(f"Training {model_name}...")
        start_time = time.time()
        
        try:
            # Create model using factory
            model = ModelFactory.create_model(model_name, num_classes)
            
            # Training parameters
            train_params = {
                'epochs': args.epochs,
                'batch': args.batch_size,
                'device': args.device,
                'project': str(out_root),
                'name': model_name.replace('.pt', ''),
                'seed': args.seed,
                **cfg.get('train_params', {})
            }
            
            # Train the model
            results = model.train(data=args.data, **train_params)
            
            # Validate the model
            val_results = model.validate(data=args.data)
            
            training_time = time.time() - start_time
            
            # Store results
            model_results = {
                'model': model_name,
                'training_time_seconds': training_time,
                'final_map50': float(val_results.box.map50) if hasattr(val_results, 'box') else None,
                'final_map50_95': float(val_results.box.map) if hasattr(val_results, 'box') else None,
            }
            
            results_summary.append(model_results)
            print(f"✓ {model_name} completed in {training_time:.1f}s")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            results_summary.append({
                'model': model_name,
                'error': str(e)
            })
    
    # Save training summary
    summary_file = out_root / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nTraining completed. Results saved to {summary_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"{'Model':<15} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'Time(s)':<10} {'Status'}")
    print("-"*80)
    
    for result in results_summary:
        if 'error' in result:
            print(f"{result['model']:<15} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'FAILED'}")
        else:
            map50 = result.get('final_map50', 0)
            map50_95 = result.get('final_map50_95', 0)
            time_s = result.get('training_time_seconds', 0)
            print(f"{result['model']:<15} {map50:<10.3f} {map50_95:<12.3f} {time_s:<10.1f} {'SUCCESS'}")

if __name__ == '__main__':
    main()
