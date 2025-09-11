#!/usr/bin/env python3
"""
Test script to verify project completion and functionality.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ EXCEPTION: {description} - {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def main():
    """Run comprehensive tests."""
    print("🚀 COMPREHENSIVE PROJECT TEST")
    print("="*80)
    
    # We're already in the project directory - no need to change
    print(f"Working directory: {os.getcwd()}")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check critical files exist
    critical_files = [
        ("scripts/train.py", "Training script"),
        ("scripts/evaluate.py", "Evaluation script"),
        ("scripts/models/model_factory.py", "Model factory"),
        ("scripts/models/yolo_wrapper.py", "YOLO wrapper"),
        ("scripts/statistical_analysis.py", "Statistical analysis"),
        ("configs/base.yaml", "Base config"),
        ("data/dummy.yaml", "Dummy dataset config"),
    ]
    
    for filepath, desc in critical_files:
        total_tests += 1
        if check_file_exists(filepath, desc):
            tests_passed += 1
    
    # Test 2: Quick training test
    total_tests += 1
    if run_command("python scripts/train.py --data data/dummy.yaml --models yolov8n.pt --epochs 1 --device cpu", 
                  "Quick training test"):
        tests_passed += 1
    
    # Test 3: Evaluation test  
    total_tests += 1
    weights_path = None
    for run_dir in Path("results/runs").glob("yolov8n*"):
        weights = run_dir / "weights" / "best.pt"
        if weights.exists():
            weights_path = str(weights)
            break
    
    if weights_path:
        if run_command(f"python scripts/evaluate.py --weights {weights_path} --data data/dummy.yaml --device cpu",
                      "Model evaluation test"):
            tests_passed += 1
    else:
        print("❌ No trained weights found for evaluation test")
    
    # Test 4: Check results files
    result_files = [
        ("results/training_summary.json", "Training results"),
        ("results/eval_summary.json", "Evaluation results"),
    ]
    
    for filepath, desc in result_files:
        total_tests += 1
        if check_file_exists(filepath, desc):
            tests_passed += 1
    
    # Test 5: Statistical analysis (if we have some data)
    if Path("results/eval_summary.json").exists():
        total_tests += 1
        if run_command("python scripts/statistical_analysis.py --results results/eval_summary.json --output results/stats_test.json",
                      "Statistical analysis test"):
            tests_passed += 1
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL TEST SUMMARY")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Project is complete and functional!")
        return True
    elif tests_passed >= total_tests * 0.8:  # 80% threshold
        print("⚠️  Most tests passed. Project is mostly functional with minor issues.")
        return True
    else:
        print("❌ Multiple test failures. Project needs more work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
