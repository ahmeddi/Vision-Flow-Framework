"""Unified training script wrapper leveraging multiple model architectures.
Supports YOLO (v8, v11, v7), YOLO-NAS, YOLOX, EfficientDet, and DETR models.
"""
import argparse, yaml, os, json, time
from pathlib import Path
import torch
import random
import numpy as np

# Import model wrappers
from models.yolo_wrapper import YOLOWrapper
# Always import YOLO-NAS wrapper (handles mock internally if super-gradients not available)
from models.yolo_nas_wrapper import YOLONASWrapper
YOLO_NAS_AVAILABLE = True

try:
    from models.yolox_wrapper import YOLOXWrapper
    YOLOX_AVAILABLE = True
except ImportError:
    YOLOX_AVAILABLE = False

try:
    from models.efficientdet_wrapper import EfficientDetWrapper
    EFFICIENTDET_AVAILABLE = True
except ImportError:
    EFFICIENTDET_AVAILABLE = False

try:
    from models.detr_wrapper import DETRWrapper
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False

# Model definitions
YOLO_MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
    'yolov7.pt'
]

YOLO_NAS_MODELS = [
    'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'
] if YOLO_NAS_AVAILABLE else []

YOLOX_MODELS = [
    'yolox_nano', 'yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
] if YOLOX_AVAILABLE else []

EFFICIENTDET_MODELS = [
    'efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 'efficientdet_d3',
    'efficientdet_d4', 'efficientdet_d5', 'efficientdet_d6', 'efficientdet_d7'
] if EFFICIENTDET_AVAILABLE else []

DETR_MODELS = [
    'detr_resnet50', 'detr_resnet101', 'rt_detr_resnet50', 'rt_detr_resnet101'
] if DETR_AVAILABLE else []

ALL_MODELS = YOLO_MODELS + YOLO_NAS_MODELS + YOLOX_MODELS + EFFICIENTDET_MODELS + DETR_MODELS

class ModelFactory:
    """Factory class to create different types of models with unified interface."""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int = 80):
        """Create model instance based on model name."""
        if model_name in YOLO_MODELS:
            return YOLOWrapper(model_name, num_classes)
        elif model_name in YOLO_NAS_MODELS and YOLO_NAS_AVAILABLE:
            return YOLONASWrapper(model_name, num_classes)
        elif model_name in YOLOX_MODELS and YOLOX_AVAILABLE:
            return YOLOXWrapper(model_name, num_classes)
        elif model_name in EFFICIENTDET_MODELS and EFFICIENTDET_AVAILABLE:
            return EfficientDetWrapper(model_name, num_classes)
        elif model_name in DETR_MODELS and DETR_AVAILABLE:
            return DETRWrapper(model_name, num_classes)
        else:
            available_models = [m for m in ALL_MODELS if m in [
                *YOLO_MODELS,
                *(YOLO_NAS_MODELS if YOLO_NAS_AVAILABLE else []),
                *(YOLOX_MODELS if YOLOX_AVAILABLE else []),
                *(EFFICIENTDET_MODELS if EFFICIENTDET_AVAILABLE else []),
                *(DETR_MODELS if DETR_AVAILABLE else [])
            ]]
            raise ValueError(f"Unsupported model: {model_name}. Available: {available_models}")

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_num_classes_from_data_yaml(data_path: str) -> int:
    """Extract number of classes from data YAML file."""
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config.get('nc', len(data_config.get('names', [])))

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Unified training script for multiple model architectures')
    parser.add_argument('--config', default='configs/base.yaml', help='Training configuration file')
    parser.add_argument('--data', required=True, help='Path to YOLO data YAML defining train/val sets')
    parser.add_argument('--models', nargs='*', default=['yolov8n.pt'], 
                       help='Models to train')
    parser.add_argument('--output', default='results/runs', help='Output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--device', default='auto', help='Training device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seeds(args.seed)

    # Load configuration
    cfg = load_config(args.config)
    num_classes = get_num_classes_from_data_yaml(args.data)
    
    # Override config with command line arguments
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    
    # Set device based on availability
    import torch
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Print available models
    if args.verbose:
        print("Available model families:")
        print(f"  YOLO: {len(YOLO_MODELS)} models")
        print(f"  YOLO-NAS: {len(YOLO_NAS_MODELS)} models" if YOLO_NAS_AVAILABLE else "  YOLO-NAS: Not available")
        print(f"  YOLOX: {len(YOLOX_MODELS)} models" if YOLOX_AVAILABLE else "  YOLOX: Not available")
        print(f"  EfficientDet: {len(EFFICIENTDET_MODELS)} models" if EFFICIENTDET_AVAILABLE else "  EfficientDet: Not available")
        print(f"  DETR: {len(DETR_MODELS)} models" if DETR_AVAILABLE else "  DETR: Not available")
        print(f"  Total available: {len(ALL_MODELS)} models")
        print()
    
    results_summary = []
    
    for model_name in args.models:
        print(f"{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        start_time = time.time()
        
        try:
            # Create model using factory
            model = ModelFactory.create_model(model_name, num_classes)
            
            # Training parameters
            train_params = {
                'epochs': args.epochs,  # Override from command line
                'batch_size': cfg.get('batch_size', 16),
                'device': device,
                'project': str(out_root),
                'name': model_name.replace('.pt', '').replace('_', '-'),
                **cfg.get('train_params', {})
            }
            
            if args.verbose:
                print(f"Training parameters: {train_params}")
            
            # Train the model
            train_results = model.train(data=args.data, **train_params)
            
            # Validate the model
            if 'best_weights' in train_results:
                val_results = model.validate(data=args.data, weights=train_results['best_weights'], device=device)
            else:
                val_results = model.validate(data=args.data, device=device)
            
            training_time = time.time() - start_time
            
            # Get model info
            model_info = model.get_model_info(train_results.get('best_weights'))
            
            # Store results
            model_results = {
                'model': model_name,
                'architecture': model_info.get('architecture', 'Unknown'),
                'training_time_seconds': training_time,
                'total_parameters': model_info.get('total_parameters', 0),
                'model_size_mb': model_info.get('model_size_mb', 0),
                'final_map50': val_results.get('map50', 0.0),
                'final_map50_95': val_results.get('map50_95', 0.0),
                'precision': val_results.get('precision', 0.0),
                'recall': val_results.get('recall', 0.0),
                'best_weights': train_results.get('best_weights', ''),
                'save_dir': train_results.get('save_dir', '')
            }
            
            results_summary.append(model_results)
            print(f"✅ {model_name} completed in {training_time:.1f}s")
            print(f"   mAP@0.5: {val_results.get('map50', 0.0):.3f}")
            print(f"   mAP@0.5:0.95: {val_results.get('map50_95', 0.0):.3f}")
            print(f"   Parameters: {model_info.get('total_parameters', 0):,}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results_summary.append({
                'model': model_name,
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            })
    
    # Save training summary
    summary_file = out_root / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Results saved to: {summary_file}")
    print()
    
    # Print summary table
    print(f"{'Model':<20} {'Architecture':<12} {'mAP@0.5':<8} {'mAP@0.5:0.95':<12} {'Params':<10} {'Time(s)':<8} {'Status'}")
    print("-"*90)
    
    for result in results_summary:
        if 'error' in result:
            print(f"{result['model']:<20} {'ERROR':<12} {'N/A':<8} {'N/A':<12} {'N/A':<10} {result.get('training_time_seconds', 0):<8.1f} {'FAILED'}")
        else:
            map50 = result.get('final_map50', 0)
            map50_95 = result.get('final_map50_95', 0)
            params = result.get('total_parameters', 0)
            arch = result.get('architecture', 'Unknown')[:11]
            time_s = result.get('training_time_seconds', 0)
            print(f"{result['model']:<20} {arch:<12} {map50:<8.3f} {map50_95:<12.3f} {params:<10,} {time_s:<8.1f} {'SUCCESS'}")
    
    print(f"\n{'='*80}")
    successful_models = len([r for r in results_summary if 'error' not in r])
    total_models = len(results_summary)
    print(f"Training completed: {successful_models}/{total_models} models successful")

if __name__ == '__main__':
    main()
