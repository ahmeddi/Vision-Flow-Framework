"""Evaluation script to compute metrics and latency for all trained models."""
import argparse, json, time, statistics
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Any

# Import model wrappers
from models.yolo_wrapper import YOLOWrapper
try:
    from models.yolo_nas_wrapper import YOLONASWrapper
    YOLO_NAS_AVAILABLE = True
except ImportError:
    YOLO_NAS_AVAILABLE = False

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

def detect_model_type(weights_path: str) -> str:
    """Detect model type from weights file path or name."""
    weights_path = weights_path.lower()
    
    if any(x in weights_path for x in ['yolov8', 'yolov11', 'yolov7']):
        return 'yolo'
    elif 'yolo_nas' in weights_path:
        return 'yolo_nas'
    elif 'yolox' in weights_path:
        return 'yolox'
    elif 'efficientdet' in weights_path:
        return 'efficientdet'
    elif any(x in weights_path for x in ['detr', 'rt_detr']):
        return 'detr'
    else:
        return 'yolo'  # Default fallback

@torch.inference_mode()
def benchmark_latency(model_wrapper, 
                     imgsz: int = 640, 
                     device: str = 'cuda', 
                     warmup: int = 10, 
                     runs: int = 50) -> Dict[str, float]:
    """Benchmark model inference latency."""
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    
    # Get the actual model for benchmarking
    if hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'predict'):
        model = model_wrapper.model
        
        # Warmup
        for _ in range(warmup):
            try:
                _ = model(dummy_input, verbose=False)
            except:
                # Fallback for models that don't accept tensor input directly
                _ = model.predict(dummy_input, verbose=False)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.time()
            try:
                _ = model(dummy_input, verbose=False)
            except:
                _ = model.predict(dummy_input, verbose=False)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            times.append(time.time() - start)
    else:
        # For models without direct tensor input, use image-based benchmarking
        import tempfile
        import cv2
        
        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        
        # Warmup
        for _ in range(min(warmup, 5)):  # Fewer warmups for file-based
            _ = model_wrapper.predict(tmp_path, verbose=False)
        
        # Benchmark
        times = []
        for _ in range(min(runs, 20)):  # Fewer runs for file-based
            start = time.time()
            _ = model_wrapper.predict(tmp_path, verbose=False)
            times.append(time.time() - start)
        
        # Cleanup
        Path(tmp_path).unlink()
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    p95_time = np.percentile(times, 95)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        'latency_ms': avg_time * 1000,
        'p95_ms': p95_time * 1000,
        'fps': fps,
        'std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0
    }

def evaluate_model(weights_path: str, 
                  data_yaml: str,
                  device: str = 'cuda',
                  imgsz: int = 640) -> Dict[str, Any]:
    """Evaluate a single model on validation set."""
    
    try:
        # Detect model type
        model_type = detect_model_type(weights_path)
        model_name = Path(weights_path).stem
        
        print(f"Evaluating {model_name} ({model_type})...")
        
        # Create model wrapper
        if model_type == 'yolo':
            model = YOLOWrapper(weights_path)  # Use full path, not just name
        elif model_type == 'yolo_nas' and YOLO_NAS_AVAILABLE:
            model = YOLONASWrapper(model_name)
        elif model_type == 'yolox' and YOLOX_AVAILABLE:
            model = YOLOXWrapper(model_name)
        elif model_type == 'efficientdet' and EFFICIENTDET_AVAILABLE:
            model = EfficientDetWrapper(model_name)
        elif model_type == 'detr' and DETR_AVAILABLE:
            model = DETRWrapper(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Validation metrics
        val_results = model.validate(data=data_yaml, weights=weights_path, device=device)
        
        # Latency benchmarking
        latency_results = benchmark_latency(model, imgsz=imgsz, device=device)
        
        # Model information
        model_info = model.get_model_info(weights_path)
        
        # Combine results
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'weights_path': weights_path,
            **val_results,
            **latency_results,
            **model_info
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {weights_path}: {e}")
        return {
            'model_name': Path(weights_path).stem,
            'weights_path': weights_path,
            'error': str(e)
        }

def find_model_weights(models_dir: str) -> List[str]:
    """Find all model weight files in directory."""
    
    models_path = Path(models_dir)
    weight_files = []
    
    # Look for common weight file patterns
    patterns = [
        '**/best.pt',
        '**/best.pth', 
        '**/weights/best.pt',
        '**/weights/best.pth',
        '**/*.pt',
        '**/*.pth'
    ]
    
    for pattern in patterns:
        weight_files.extend(models_path.glob(pattern))
    
    # Remove duplicates and sort
    weight_files = sorted(list(set(weight_files)))
    
    return [str(f) for f in weight_files]

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--models_dir', default='results/runs', 
                       help='Directory containing trained models')
    parser.add_argument('--data', help='Data YAML file for validation')
    parser.add_argument('--weights', nargs='*', help='Specific weight files to evaluate')
    parser.add_argument('--device', default='cuda', help='Device for evaluation')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--output', default='results/eval_summary.json', 
                       help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Find weight files
    if args.weights:
        weight_files = args.weights
    else:
        weight_files = find_model_weights(args.models_dir)
    
    if not weight_files:
        print(f"No weight files found in {args.models_dir}")
        return
    
    print(f"Found {len(weight_files)} model weight files")
    if args.verbose:
        for w in weight_files:
            print(f"  {w}")
    
    # Find data YAML file if not specified
    if not args.data:
        # Look for data.yaml in models directory or parent
        models_path = Path(args.models_dir)
        potential_data_files = [
            models_path / 'data.yaml',
            models_path.parent / 'data.yaml',
            Path('data/sample_weeds.yaml'),
            Path('data/dummy.yaml')
        ]
        
        for data_file in potential_data_files:
            if data_file.exists():
                args.data = str(data_file)
                break
        
        if not args.data:
            print("No data.yaml file found. Please specify with --data")
            return
    
    print(f"Using data file: {args.data}")
    
    # Evaluate all models
    all_results = []
    successful = 0
    
    for i, weights_path in enumerate(weight_files):
        print(f"\n[{i+1}/{len(weight_files)}] {Path(weights_path).name}")
        
        results = evaluate_model(
            weights_path=weights_path,
            data_yaml=args.data,
            device=args.device,
            imgsz=args.imgsz
        )
        
        all_results.append(results)
        
        if 'error' not in results:
            successful += 1
            print(f"  ✅ mAP@0.5: {results.get('map50', 0):.3f}, "
                  f"FPS: {results.get('fps', 0):.1f}, "
                  f"Params: {results.get('total_parameters', 0):,}")
        else:
            print(f"  ❌ {results['error']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Evaluated: {successful}/{len(weight_files)} models")
    print(f"Results saved to: {output_path}")
    
    # Print summary table
    if successful > 0:
        print(f"\n{'Model':<20} {'Type':<12} {'mAP@0.5':<8} {'FPS':<8} {'Latency':<10} {'Params':<10}")
        print("-" * 80)
        
        for result in all_results:
            if 'error' not in result:
                name = result.get('model_name', 'Unknown')[:19]
                model_type = result.get('model_type', 'Unknown')[:11] 
                map50 = result.get('map50', 0)
                fps = result.get('fps', 0)
                latency = result.get('latency_ms', 0)
                params = result.get('total_parameters', 0)
                
                print(f"{name:<20} {model_type:<12} {map50:<8.3f} {fps:<8.1f} "
                      f"{latency:<10.1f} {params:<10,}")

if __name__ == '__main__':
    main()
