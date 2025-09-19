#!/usr/bin/env python3
"""
VFF Colab Setup Script
=====================
One-command setup for running Vision Flow Framework on Google Colab.
"""

import subprocess
import sys
import os
from pathlib import Path


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def run_command(cmd, description, capture_output=False):
    """Run a command and report success/failure."""
    print(f"\n🔄 {description}...")
    print(f"   Command: {cmd}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=True)
        
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        return False


def setup_colab_environment():
    """Setup VFF environment on Google Colab."""
    print("🚀 Setting up Vision Flow Framework on Google Colab")
    print("=" * 60)
    
    # Check if in Colab
    if not is_colab():
        print("⚠️  This appears to be a local environment, not Colab")
        print("   You can still run this script, but some steps may differ")
    
    # Clone repository if not already present
    if not Path('/content/vff').exists() and not Path('scripts/train.py').exists():
        print("\n📂 Cloning VFF repository...")
        run_command("git clone https://github.com/ahmeddi/Vision-Flow-Framework.git /content/vff", 
                   "Cloning repository")
        os.chdir('/content/vff')
    
    # Show current directory
    current_dir = Path.cwd()
    print(f"\n📍 Working directory: {current_dir}")
    
    # Install core dependencies
    print("\n📦 Installing dependencies...")
    deps = [
        "ultralytics",
        "torch torchvision",
        "opencv-python",
        "Pillow",
        "requests tqdm pyyaml",
        "pandas numpy",
        "matplotlib seaborn plotly",
        "scikit-learn scipy"
    ]
    
    for dep in deps:
        run_command(f"pip install -q {dep}", f"Installing {dep}")
    
    # Try to install optional dependencies
    print("\n🔧 Installing optional architectures...")
    
    # Handle Colab-specific issues first
    if is_colab():
        print("   Setting up Colab environment...")
        try:
            run_command("apt-get update -qq", "Updating system packages")
            run_command("apt-get install -y -qq gcc g++", "Installing build tools")
        except:
            print("   ⚠️ System package setup skipped")
    
    # Install pycocotools with error handling
    try:
        print("   Installing pycocotools...")
        success = run_command("pip install -q pycocotools-windows", "pycocotools (Windows)")
        if not success:
            run_command("pip install -q pycocotools", "pycocotools (generic)")
    except:
        print("   ⚠️ pycocotools installation failed (some models may not work)")
    
    # Install advanced architectures with better error handling
    optional_deps = {
        "super-gradients --no-deps": "YOLO-NAS core",
        "omegaconf hydra-core": "YOLO-NAS dependencies", 
        "timm": "EfficientDet support",
        "transformers": "DETR support"
    }
    
    for dep, description in optional_deps.items():
        try:
            run_command(f"pip install -q {dep}", f"Installing {description}")
        except:
            print(f"⚠️  {description} installation skipped (optional)")
    
    # Download essential models
    print("\n🎯 Downloading models...")
    run_command("python scripts/download_models.py --set essential", 
               "Downloading essential models")
    
    # Generate dummy dataset
    print("\n📊 Creating sample dataset...")
    run_command("python scripts/generate_dummy_data.py --n_train 50 --n_val 20", 
               "Creating dummy dataset")
    
    # Verify setup
    print("\n🔍 Verifying setup...")
    run_command("python test_models_availability.py", "Testing setup")
    
    # Quick training test
    print("\n🧪 Running quick training test...")
    run_command("python scripts/train.py --models yolov8n.pt --data data/dummy.yaml --epochs 1 --batch-size 8", 
               "Quick training test")
    
    print("\n" + "=" * 60)
    print("🎉 COLAB SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n🚀 Quick start commands for Colab:")
    print("   # Quick comparison")
    print("   !python scripts/train.py --models yolov8n.pt yolo11n.pt --data data/dummy.yaml --epochs 3")
    print()
    print("   # Download more models")
    print("   !python scripts/download_models.py --set research")
    print()
    print("   # Download real datasets")
    print("   !python scripts/download_datasets.py --datasets sample_weeds --sample 30")
    print()
    print("   # Full research study")
    print("   !python scripts/run_comprehensive_training.py")
    print()
    print("   # Generate visualizations")
    print("   !python scripts/create_visualizations.py")
    
    print("\n💾 Download results:")
    print("   from google.colab import files")
    print("   files.download('results.zip')")
    
    return True


def create_colab_commands():
    """Generate useful Colab commands for users."""
    commands = {
        "Basic Training": [
            "# Quick test (1 epoch)",
            "!python scripts/train.py --models yolov8n.pt --data data/dummy.yaml --epochs 1",
            "",
            "# Compare models (3 epochs)", 
            "!python scripts/train.py --models yolov8n.pt yolo11n.pt --data data/dummy.yaml --epochs 3",
        ],
        
        "Download More": [
            "# See available models",
            "!python scripts/download_models.py --list",
            "",
            "# Download research models",
            "!python scripts/download_models.py --set research",
            "",
            "# Download real datasets",
            "!python scripts/download_datasets.py --datasets sample_weeds --sample 50",
        ],
        
        "Advanced Experiments": [
            "# Full comparison study", 
            "!python scripts/run_comprehensive_training.py",
            "",
            "# Custom training",
            "!python scripts/train.py --models yolov8s.pt --data data/sample_weeds.yaml --epochs 10",
            "",
            "# Generate analysis",
            "!python scripts/create_visualizations.py",
        ],
        
        "Results & Download": [
            "# Create results archive",
            "!zip -r results.zip results/ *.pt",
            "",
            "# Download results",
            "from google.colab import files",
            "files.download('results.zip')",
            "",
            "# Show training plots",
            "import matplotlib.pyplot as plt",
            "import matplotlib.image as mpimg",
            "from pathlib import Path",
            "",
            "plots = list(Path('results').rglob('*.png'))",
            "for plot in plots[:3]:",
            "    plt.figure(figsize=(10,6))", 
            "    img = mpimg.imread(plot)",
            "    plt.imshow(img)",
            "    plt.axis('off')",
            "    plt.title(plot.name)",
            "    plt.show()",
        ]
    }
    
    print("\n📋 USEFUL COLAB COMMANDS")
    print("=" * 50)
    
    for category, cmd_list in commands.items():
        print(f"\n🔹 {category}:")
        for cmd in cmd_list:
            print(f"   {cmd}")


def main():
    """Main function for Colab setup."""
    print("🚀 VFF Colab Setup")
    print("=" * 30)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--commands':
        create_colab_commands()
        return
    
    try:
        setup_colab_environment()
        create_colab_commands()
        
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        print("\n🔧 Manual setup steps:")
        print("1. Install dependencies: !pip install ultralytics torch opencv-python")
        print("2. Download models: !python scripts/download_models.py --set essential")
        print("3. Create dataset: !python scripts/generate_dummy_data.py")
        print("4. Test training: !python scripts/train.py --models yolov8n.pt --data data/dummy.yaml --epochs 1")


if __name__ == '__main__':
    main()