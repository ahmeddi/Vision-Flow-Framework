#!/usr/bin/env python3
"""Test script to check model availability and verify downloaded models."""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(__file__))

def check_model_files():
    """Check for downloaded model files in the project."""
    print("🔍 Checking for downloaded model files...")
    
    # Look for .pt files in common locations
    search_paths = [
        Path.cwd(),  # Project root
        Path.cwd() / 'models',  # Models directory
        Path.cwd() / 'weights',  # Weights directory
    ]
    
    found_models = []
    for search_path in search_paths:
        if search_path.exists():
            pt_files = list(search_path.glob('*.pt'))
            for pt_file in pt_files:
                found_models.append(pt_file)
    
    if found_models:
        print(f"✅ Found {len(found_models)} model files:")
        for model in found_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   📄 {model.name} ({size_mb:.1f} MB) - {model.parent}")
    else:
        print("❌ No model files found!")
        print("   💡 Download models with: python scripts/download_models.py --set essential")
    
    return found_models


def verify_model_loading():
    """Test loading models with PyTorch."""
    print("\n🧪 Testing model loading...")
    
    try:
        import torch
        print("✅ PyTorch is available")
    except ImportError:
        print("❌ PyTorch not available - cannot test model loading")
        return False
    
    # Check for models to test
    model_files = [
        'yolov8n.pt',
        'yolo11n.pt',
        'yolov8s.pt',
        'yolo11s.pt'
    ]
    
    loaded_count = 0
    for model_name in model_files:
        model_path = Path(model_name)
        if model_path.exists():
            try:
                # Try to load the model
                model = torch.load(model_path, map_location='cpu')
                print(f"✅ {model_name} - loads successfully")
                loaded_count += 1
            except Exception as e:
                print(f"❌ {model_name} - failed to load: {e}")
        else:
            print(f"⚪ {model_name} - not found")
    
    return loaded_count > 0


def test_ultralytics_integration():
    """Test if Ultralytics can access the models."""
    print("\n🔧 Testing Ultralytics integration...")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics is available")
        
        # Test loading models
        test_models = ['yolov8n.pt', 'yolo11n.pt']
        for model_name in test_models:
            if Path(model_name).exists():
                try:
                    model = YOLO(model_name)
                    print(f"✅ {model_name} - Ultralytics can load")
                except Exception as e:
                    print(f"❌ {model_name} - Ultralytics failed: {e}")
            else:
                print(f"⚪ {model_name} - not found")
                
    except ImportError:
        print("❌ Ultralytics not available")
        print("   💡 Install with: pip install ultralytics")
        return False
    
    return True


def test_model_factory():
    """Test the project's model factory."""
    print("\n🏭 Testing Model Factory...")
    
    try:
        from scripts.train import ModelFactory
        
        # Test getting available models
        try:
            supported = ModelFactory.get_supported_models()
            print(f"✅ Model Factory supports {len(supported)} model types")
            
            # Show first few supported models
            print("   Supported architectures:")
            for model in supported[:8]:
                print(f"      • {model}")
            if len(supported) > 8:
                print(f"      ... and {len(supported) - 8} more")
                
        except Exception as e:
            print(f"❌ Model Factory error: {e}")
            
    except ImportError as e:
        print(f"❌ Cannot import Model Factory: {e}")
        return False
    
    return True


def print_setup_instructions():
    """Print helpful setup instructions."""
    print("\n" + "="*60)
    print("📋 SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n🚀 To get started quickly:")
    print("   python setup_vff.py")
    
    print("\n📦 To download specific models:")
    print("   python scripts/download_models.py --set essential")
    print("   python scripts/download_models.py --list")
    
    print("\n🧪 To test training:")
    print("   python scripts/train.py --models yolov8n.pt --data data/dummy.yaml --epochs 1")
    
    print("\n📚 For more information:")
    print("   • Check README.md")
    print("   • Run: python scripts/download_models.py --help")


def main():
    """Run all model availability tests."""
    print("🔬 VFF Model Availability Test")
    print("="*50)
    
    # Run all tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Check for model files
    found_models = check_model_files()
    if found_models:
        tests_passed += 1
    
    # Test 2: Verify model loading
    if verify_model_loading():
        tests_passed += 1
    
    # Test 3: Test Ultralytics integration
    if test_ultralytics_integration():
        tests_passed += 1
    
    # Test 4: Test Model Factory
    if test_model_factory():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 3:
        print("🎉 VFF is ready to use!")
        print("   You can start training models now.")
    elif tests_passed >= 2:
        print("⚠️  VFF is partially ready.")
        print("   Some features may not work correctly.")
    else:
        print("❌ VFF setup incomplete.")
        print("   Please follow the setup instructions below.")
    
    if tests_passed < total_tests:
        print_setup_instructions()


if __name__ == "__main__":
    main()