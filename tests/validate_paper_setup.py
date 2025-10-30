"""Quick setup and validation script for paper requirements.
Tests all new model implementations and dataset configurations.
"""
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

def test_model_factory():
    """Test updated model factory with new models."""
    print("🧪 Testing Model Factory...")
    try:
        from scripts.models.model_factory import ModelFactory, get_available_models, list_supported_models
        
        # Check available model types
        available = get_available_models()
        print("\n📋 Available Model Types:")
        for model_type, is_available in available.items():
            status = "✅" if is_available else "❌"
            print(f"  {status} {model_type}")
        
        # List supported models
        models = list_supported_models()
        print(f"\n🎯 Total Supported Models: {len(models)}")
        
        # Test creating YOLO model
        if available['yolo']:
            try:
                model = ModelFactory.create_model('yolov8n.pt')
                print("✅ YOLO model creation successful")
            except Exception as e:
                print(f"❌ YOLO model creation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False


def test_dataset_configs():
    """Test all weed dataset configurations."""
    print("\n🌱 Testing Dataset Configurations...")
    
    datasets = ['deepweeds.yaml', 'weed25.yaml', 'cwd30.yaml', 'weedsgalore.yaml']
    
    for dataset in datasets:
        dataset_path = Path('data') / dataset
        if dataset_path.exists():
            print(f"✅ {dataset} - Configuration exists")
            
            # Test YAML parsing
            try:
                import yaml
                with open(dataset_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                required_keys = ['path', 'train', 'val', 'nc', 'names']
                missing_keys = [key for key in required_keys if key not in config]
                
                if not missing_keys:
                    print(f"  ✅ Valid configuration ({config['nc']} classes)")
                else:
                    print(f"  ⚠️ Missing keys: {missing_keys}")
                    
            except Exception as e:
                print(f"  ❌ Invalid YAML: {e}")
        else:
            print(f"❌ {dataset} - Missing configuration")


def test_synthetic_data_generation():
    """Test synthetic dataset generation."""
    print("\n🔬 Testing Synthetic Data Generation...")
    
    try:
        import subprocess
        import sys
        
        # Test dataset download script
        result = subprocess.run([
            sys.executable, 'scripts/download_weed_datasets.py', 
            '--datasets', 'deepweeds', 
            '--sample', '5',
            '--output', 'data'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Synthetic data generation successful")
            
            # Check if files were created
            deepweeds_dir = Path('data/deepweeds')
            if deepweeds_dir.exists():
                train_images = list((deepweeds_dir / 'images' / 'train').glob('*.jpg'))
                train_labels = list((deepweeds_dir / 'labels' / 'train').glob('*.txt'))
                
                print(f"  ✅ Created {len(train_images)} training images")
                print(f"  ✅ Created {len(train_labels)} training labels")
            else:
                print("  ⚠️ Dataset directory not found")
        else:
            print(f"❌ Synthetic data generation failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")


def test_paper_pipeline():
    """Test end-to-end pipeline for paper."""
    print("\n📄 Testing Paper Pipeline...")
    
    try:
        import subprocess
        import sys
        
        # Quick training test
        print("  🏃 Quick training test...")
        result = subprocess.run([
            sys.executable, 'scripts/train.py',
            '--data', 'data/dummy.yaml',  # Use existing dummy data
            '--models', 'yolov8n.pt',
            '--epochs', '1',
            '--batch-size', '2'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("  ✅ Training pipeline works")
            
            # Test evaluation
            print("  📊 Testing evaluation...")
            eval_result = subprocess.run([
                sys.executable, 'scripts/evaluate.py',
                '--models_dir', 'results/runs'
            ], capture_output=True, text=True, timeout=60)
            
            if eval_result.returncode == 0:
                print("  ✅ Evaluation pipeline works")
            else:
                print(f"  ❌ Evaluation failed: {eval_result.stderr}")
        else:
            print(f"  ❌ Training failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")


def check_dependencies():
    """Check required dependencies for paper models."""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'YOLO models'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn')
    ]
    
    optional_packages = [
        ('super_gradients', 'YOLO-NAS'),
        ('yolox', 'YOLOX'),
        ('transformers', 'DETR models'),
        ('timm', 'EfficientDet backbones')
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
    
    print("\n📦 Optional Dependencies:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} - Optional (some models may not work)")


def main():
    """Run all validation tests."""
    print("🚀 Paper Requirements Validation")
    print("=" * 50)
    
    # Run all tests
    check_dependencies()
    test_model_factory()
    test_dataset_configs()
    test_synthetic_data_generation()
    test_paper_pipeline()
    
    print("\n" + "=" * 50)
    print("🎉 Validation Complete!")
    print("\n📝 Next Steps:")
    print("1. Install missing dependencies if any")
    print("2. Run comprehensive benchmark:")
    print("   python scripts/comprehensive_benchmark.py")
    print("3. Generate paper figures:")
    print("   python scripts/generate_paper_figures.py")
    print("4. Acquire real weed datasets for publication")


if __name__ == "__main__":
    main()
