#!/usr/bin/env python3
"""Test script to debug YOLOv11 issue."""

import sys
sys.path.append('scripts')

from train import YOLO_MODELS, ModelFactory

def test_yolo11():
    print("Testing YOLOv11 model support...")
    
    target = 'yolo11n.pt'
    print(f"Target model: {repr(target)}")
    print(f"YOLO_MODELS list: {YOLO_MODELS}")
    
    # Test membership
    print(f"'{target}' in YOLO_MODELS: {target in YOLO_MODELS}")
    
    # Find similar models
    for model in YOLO_MODELS:
        if 'yolo11' in model:
            print(f"Found similar: {repr(model)}")
            print(f"  Equal to target: {model == target}")
            print(f"  Char comparison: {[ord(c) for c in model]} vs {[ord(c) for c in target]}")
    
    # Test ModelFactory
    try:
        model = ModelFactory.create_model(target, 80)
        print("✅ ModelFactory.create_model() succeeded")
        return True
    except Exception as e:
        print(f"❌ ModelFactory.create_model() failed: {e}")
        return False

if __name__ == "__main__":
    success = test_yolo11()
    print(f"Test result: {'PASS' if success else 'FAIL'}")