#!/usr/bin/env python3
"""
VFF Quick Setup Script
=====================
One-command setup for Vision Flow Framework.
Downloads essential models and datasets for immediate use.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"\n🔄 {description}...")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stderr:
            print(f"   Details: {e.stderr}")
        return False


def check_python_packages():
    """Check if required packages are installed."""
    print("🔍 Checking Python packages...")
    
    required_packages = ['torch', 'ultralytics', 'requests', 'tqdm', 'pyyaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True


def main():
    print("🚀 VFF (Vision Flow Framework) Quick Setup")
    print("=" * 50)
    print("This script will:")
    print("  📦 Download essential pre-trained models")
    print("  📊 Download sample datasets")
    print("  ✅ Verify everything is ready to use")
    print()
    
    # Check if we're in the right directory
    if not (Path.cwd() / 'scripts' / 'train.py').exists():
        print("❌ Please run this script from the VFF project root directory")
        print("   (where you can see the 'scripts' folder)")
        sys.exit(1)
    
    # Check Python packages
    if not check_python_packages():
        print("\n❌ Please install required packages first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    success_count = 0
    
    # Download essential models
    if run_command("python scripts/download_models.py --set essential", "Downloading essential models"):
        success_count += 1
    
    # Download sample dataset
    if run_command("python scripts/download_datasets.py --datasets sample_weeds --sample 10", "Downloading sample dataset"):
        success_count += 1
    
    # Run a quick test
    if run_command("python scripts/test_models_availability.py", "Testing model availability"):
        success_count += 1
    
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    if success_count == 3:
        print("🎉 SUCCESS! VFF is ready to use!")
        print("\n🚀 Quick start commands:")
        print("   # Test with essential models")
        print("   python scripts/train.py --models yolov8n.pt yolo11n.pt --data data/sample_weeds.yaml --epochs 1")
        print()
        print("   # Download more models")
        print("   python scripts/download_models.py --list")
        print("   python scripts/download_models.py --set research")
        print()
        print("   # Download more datasets")
        print("   python scripts/download_datasets.py --datasets deepweeds --sample 100")
        print()
        print("   # Run comprehensive training")
        print("   python scripts/run_comprehensive_training.py")
        
    else:
        print(f"⚠️  Setup partially completed ({success_count}/3 steps)")
        print("   Some components may not work correctly.")
        print("   Check the error messages above and try running the failed commands manually.")
    
    print(f"\n📁 Project structure:")
    print(f"   📂 {Path.cwd()}")
    print(f"   ├── 📄 yolov8n.pt, yolo11n.pt (models)")
    print(f"   ├── 📂 data/ (datasets)")
    print(f"   ├── 📂 scripts/ (training scripts)")
    print(f"   └── 📂 results/ (training results)")


if __name__ == '__main__':
    main()