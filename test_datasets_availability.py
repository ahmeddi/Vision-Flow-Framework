#!/usr/bin/env python3
"""Test script to check dataset availability."""

import sys
import os
from pathlib import Path

def check_dataset_availability():
    """Check availability of all datasets."""
    print("=== Test de Disponibilité des Datasets ===\n")
    
    base_path = Path("data")
    datasets = ["deepweeds", "weed25", "cwd30", "weedsgalore"]
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        images_path = dataset_path / "images"
        labels_path = dataset_path / "labels"
        
        print(f"📁 {dataset.upper()}:")
        print(f"  Path: {dataset_path}")
        print(f"  Exists: {'✓' if dataset_path.exists() else '✗'}")
        
        if images_path.exists():
            # Count images recursively
            image_files = list(images_path.rglob("*.jpg")) + list(images_path.rglob("*.png")) + list(images_path.rglob("*.jpeg"))
            print(f"  Images: {len(image_files)} files")
            
            # Check for train/val/test splits
            train_path = images_path / "train"
            val_path = images_path / "val"
            test_path = images_path / "test"
            
            if train_path.exists():
                train_images = list(train_path.rglob("*.jpg")) + list(train_path.rglob("*.png"))
                print(f"    - Train: {len(train_images)} images")
            
            if val_path.exists():
                val_images = list(val_path.rglob("*.jpg")) + list(val_path.rglob("*.png"))
                print(f"    - Val: {len(val_images)} images")
                
            if test_path.exists():
                test_images = list(test_path.rglob("*.jpg")) + list(test_path.rglob("*.png"))
                print(f"    - Test: {len(test_images)} images")
        else:
            print(f"  Images: ✗ (no images directory)")
        
        if labels_path.exists():
            label_files = list(labels_path.rglob("*.txt"))
            print(f"  Labels: {len(label_files)} files")
        else:
            print(f"  Labels: ✗ (no labels directory)")
        
        print()

if __name__ == "__main__":
    check_dataset_availability()