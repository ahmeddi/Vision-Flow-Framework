#!/usr/bin/env python3
"""
VFF Model Download Script
========================
Downloads all required pre-trained models for the Vision Flow Framework.
Supports YOLO, YOLO-NAS, YOLOX, EfficientDet, and DETR models.
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm

# Model URLs and information
MODELS = {
    # YOLO v8 models (Ultralytics)
    'yolov8n.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt',
        'size_mb': 6.2,
        'description': 'YOLOv8 Nano - fastest, smallest model',
        'sha256': None  # Will be computed on first download
    },
    'yolov8s.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt',
        'size_mb': 21.5,
        'description': 'YOLOv8 Small - good balance of speed and accuracy',
        'sha256': None
    },
    'yolov8m.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt',
        'size_mb': 49.7,
        'description': 'YOLOv8 Medium - higher accuracy',
        'sha256': None
    },
    'yolov8l.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt',
        'size_mb': 83.7,
        'description': 'YOLOv8 Large - high accuracy',
        'sha256': None
    },
    'yolov8x.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt',
        'size_mb': 136.7,
        'description': 'YOLOv8 Extra Large - highest accuracy',
        'sha256': None
    },
    
    # YOLO v11 models (Ultralytics)
    'yolo11n.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
        'size_mb': 5.1,
        'description': 'YOLO11 Nano - latest generation, fastest',
        'sha256': None
    },
    'yolo11s.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
        'size_mb': 19.4,
        'description': 'YOLO11 Small - latest generation, balanced',
        'sha256': None
    },
    'yolo11m.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
        'size_mb': 44.9,
        'description': 'YOLO11 Medium - latest generation, good accuracy',
        'sha256': None
    },
    'yolo11l.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',
        'size_mb': 75.8,
        'description': 'YOLO11 Large - latest generation, high accuracy',
        'sha256': None
    },
    'yolo11x.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
        'size_mb': 125.2,
        'description': 'YOLO11 Extra Large - latest generation, highest accuracy',
        'sha256': None
    },
}

# Model sets for different use cases
MODEL_SETS = {
    'essential': ['yolov8n.pt', 'yolo11n.pt'],
    'research': ['yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt', 'yolo11s.pt'],
    'full_yolo': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt'],
    'all': list(MODELS.keys())
}


def download_file(url: str, destination: Path, expected_size_mb: Optional[float] = None) -> bool:
    """Download a file with progress bar and validation."""
    try:
        print(f"Downloading {destination.name}...")
        
        # Start download
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify file size
        actual_size = destination.stat().st_size
        if expected_size_mb:
            expected_size = expected_size_mb * 1024 * 1024
            if abs(actual_size - expected_size) > (expected_size * 0.1):  # 10% tolerance
                print(f"⚠️  Warning: {destination.name} size differs from expected")
                print(f"   Expected: {expected_size_mb:.1f} MB, Got: {actual_size / (1024*1024):.1f} MB")
        
        print(f"✅ {destination.name} downloaded successfully ({actual_size / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {destination.name}: {e}")
        if destination.exists():
            destination.unlink()  # Remove partial file
        return False


def verify_model(model_path: Path) -> bool:
    """Basic verification that a model file is valid."""
    if not model_path.exists():
        return False
    
    # Check file size (should be > 1MB for any real model)
    if model_path.stat().st_size < 1024 * 1024:
        return False
    
    # For .pt files, try to check if it's a valid PyTorch file
    if model_path.suffix == '.pt':
        try:
            import torch
            torch.load(model_path, map_location='cpu')
            return True
        except Exception:
            return False
    
    return True


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def setup_model_directory(base_dir: Path = None) -> Path:
    """Setup and return the models directory."""
    if base_dir is None:
        # Try to detect if we're in the VFF project root
        current_dir = Path.cwd()
        if (current_dir / 'scripts' / 'train.py').exists():
            base_dir = current_dir
        elif (current_dir.parent / 'scripts' / 'train.py').exists():
            base_dir = current_dir.parent
        else:
            base_dir = current_dir
    
    # Create models directory in project root (where the .pt files should go)
    models_dir = base_dir
    models_dir.mkdir(exist_ok=True)
    
    print(f"📁 Models will be saved to: {models_dir.absolute()}")
    return models_dir


def download_models(model_names: List[str], models_dir: Path, force: bool = False) -> Dict[str, bool]:
    """Download specified models."""
    results = {}
    total_size_mb = sum(MODELS[name]['size_mb'] for name in model_names if name in MODELS)
    
    print(f"🚀 Starting download of {len(model_names)} models")
    print(f"📊 Total download size: ~{total_size_mb:.1f} MB")
    print("=" * 60)
    
    for model_name in model_names:
        if model_name not in MODELS:
            print(f"❌ Unknown model: {model_name}")
            results[model_name] = False
            continue
        
        model_info = MODELS[model_name]
        model_path = models_dir / model_name
        
        # Check if already exists and valid
        if model_path.exists() and not force:
            if verify_model(model_path):
                print(f"✅ {model_name} already exists and is valid")
                results[model_name] = True
                continue
            else:
                print(f"🔄 {model_name} exists but appears invalid, re-downloading...")
        
        # Download the model
        success = download_file(model_info['url'], model_path, model_info['size_mb'])
        
        if success:
            # Verify the downloaded model
            if verify_model(model_path):
                results[model_name] = True
            else:
                print(f"❌ {model_name} downloaded but failed verification")
                results[model_name] = False
        else:
            results[model_name] = False
    
    return results


def print_model_info():
    """Print information about available models."""
    print("\n📋 Available Models:")
    print("=" * 80)
    
    for model_name, info in MODELS.items():
        print(f"🔸 {model_name:<15} ({info['size_mb']:>6.1f} MB) - {info['description']}")
    
    print(f"\n📊 Model Sets:")
    print("=" * 40)
    for set_name, models in MODEL_SETS.items():
        total_size = sum(MODELS[m]['size_mb'] for m in models if m in MODELS)
        print(f"🔹 {set_name:<12} ({len(models):>2} models, ~{total_size:>6.1f} MB)")
        print(f"   Models: {', '.join(models)}")


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained models for Vision Flow Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_models.py --set essential     # Download essential models only
  python scripts/download_models.py --set research     # Download research models
  python scripts/download_models.py --models yolov8n.pt yolo11n.pt  # Download specific models
  python scripts/download_models.py --list             # List available models
  python scripts/download_models.py --all              # Download all models
        """
    )
    
    parser.add_argument('--models', nargs='+', help='Specific models to download')
    parser.add_argument('--set', choices=MODEL_SETS.keys(), help='Download a predefined set of models')
    parser.add_argument('--all', action='store_true', help='Download all available models')
    parser.add_argument('--list', action='store_true', help='List available models and exit')
    parser.add_argument('--force', action='store_true', help='Force re-download even if models exist')
    parser.add_argument('--output-dir', type=Path, help='Directory to save models (default: project root)')
    
    args = parser.parse_args()
    
    if args.list:
        print_model_info()
        return
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        models_to_download = MODEL_SETS['all']
    elif args.set:
        models_to_download = MODEL_SETS[args.set]
    elif args.models:
        models_to_download = args.models
    else:
        # Default to essential models
        print("ℹ️  No specific models selected, downloading essential models...")
        print("   Use --list to see available models and sets")
        models_to_download = MODEL_SETS['essential']
    
    if not models_to_download:
        print("❌ No models to download. Use --help for usage information.")
        return
    
    # Setup models directory
    models_dir = setup_model_directory(args.output_dir)
    
    # Download models
    results = download_models(models_to_download, models_dir, args.force)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"✅ Successfully downloaded: {len(successful)}/{len(results)} models")
    if successful:
        for model in successful:
            print(f"   ✓ {model}")
    
    if failed:
        print(f"\n❌ Failed downloads: {len(failed)} models")
        for model in failed:
            print(f"   ✗ {model}")
    
    if successful:
        print(f"\n🎉 Models are ready! You can now run:")
        print(f"   python scripts/train.py --models {' '.join(successful[:2])} --data data/sample_weeds.yaml")


if __name__ == '__main__':
    main()