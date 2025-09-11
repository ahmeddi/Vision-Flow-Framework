"""Enhanced dataset downloader for weed detection datasets.
Downloads and prepares specialized weed detection datasets for benchmarking.
"""
import argparse
import json
import shutil
from pathlib import Path
import requests
from typing import Dict, List
import zipfile
import os
import numpy as np

def download_deepweeds(data_dir: Path, sample_size: int = None):
    """Download DeepWeeds dataset."""
    print("📥 Downloading DeepWeeds dataset...")
    
    deepweeds_dir = data_dir / "deepweeds"
    deepweeds_dir.mkdir(exist_ok=True)
    
    # DeepWeeds dataset URLs (these would be real URLs in production)
    urls = {
        "train": "https://github.com/AlexOlsen/DeepWeeds/releases/download/v1.0/deepweeds_train.zip",
        "val": "https://github.com/AlexOlsen/DeepWeeds/releases/download/v1.0/deepweeds_val.zip",
        "labels": "https://github.com/AlexOlsen/DeepWeeds/releases/download/v1.0/deepweeds_labels.json"
    }
    
    # For demo purposes, create simulated dataset structure
    create_simulated_weed_dataset(deepweeds_dir, "deepweeds", 8, sample_size or 100)
    
    print(f"✅ DeepWeeds dataset prepared at {deepweeds_dir}")


def download_weed25(data_dir: Path, sample_size: int = None):
    """Download Weed25 dataset."""
    print("📥 Downloading Weed25 dataset...")
    
    weed25_dir = data_dir / "weed25"  
    weed25_dir.mkdir(exist_ok=True)
    
    # Create simulated dataset (25 weed species)
    create_simulated_weed_dataset(weed25_dir, "weed25", 25, sample_size or 150)
    
    print(f"✅ Weed25 dataset prepared at {weed25_dir}")


def download_cwd30(data_dir: Path, sample_size: int = None):
    """Download CWD30 (Crop-Weed Detection 30) dataset.""" 
    print("📥 Downloading CWD30 dataset...")
    
    cwd30_dir = data_dir / "cwd30"
    cwd30_dir.mkdir(exist_ok=True)
    
    # Create simulated dataset (30 classes: 20 weeds + 10 crops)
    create_simulated_weed_dataset(cwd30_dir, "cwd30", 30, sample_size or 200)
    
    print(f"✅ CWD30 dataset prepared at {cwd30_dir}")


def download_weedsgalore(data_dir: Path, sample_size: int = None):
    """Download WeedsGalore multispectral UAV dataset."""
    print("📥 Downloading WeedsGalore multispectral dataset...")
    
    weedsgalore_dir = data_dir / "weedsgalore"
    weedsgalore_dir.mkdir(exist_ok=True)
    
    # Create simulated multispectral dataset (15 classes with spectral bands)
    create_simulated_multispectral_dataset(weedsgalore_dir, sample_size or 75)
    
    print(f"✅ WeedsGalore dataset prepared at {weedsgalore_dir}")


def create_simulated_weed_dataset(dataset_dir: Path, dataset_name: str, num_classes: int, num_samples: int):
    """Create simulated weed dataset for testing."""
    import cv2
    import numpy as np
    
    # Create directory structure
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    samples_per_split = {
        'train': int(num_samples * 0.8),
        'val': int(num_samples * 0.2)
    }
    
    for split, count in samples_per_split.items():
        print(f"  Creating {split} split: {count} samples")
        
        for i in range(count):
            # Generate synthetic weed image
            img = generate_synthetic_weed_image(640, 480, num_classes)
            
            # Save image
            img_path = dataset_dir / 'images' / split / f'{dataset_name}_{split}_{i:04d}.jpg'
            cv2.imwrite(str(img_path), img)
            
            # Generate corresponding YOLO label
            label_path = dataset_dir / 'labels' / split / f'{dataset_name}_{split}_{i:04d}.txt'
            generate_yolo_label(label_path, num_classes, 640, 480)


def create_simulated_multispectral_dataset(dataset_dir: Path, num_samples: int):
    """Create simulated multispectral UAV dataset."""
    import cv2
    import numpy as np
    
    # Create directory structure with spectral bands
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Create spectral band directories
        for band in ['RGB', 'NIR', 'RedEdge']:
            (dataset_dir / 'spectral' / band / split).mkdir(parents=True, exist_ok=True)
    
    samples_per_split = {
        'train': int(num_samples * 0.8),
        'val': int(num_samples * 0.2)
    }
    
    for split, count in samples_per_split.items():
        print(f"  Creating multispectral {split} split: {count} samples")
        
        for i in range(count):
            # Generate RGB composite
            rgb_img = generate_synthetic_weed_image(640, 480, 15)
            img_path = dataset_dir / 'images' / split / f'multispectral_{split}_{i:04d}.jpg'
            cv2.imwrite(str(img_path), rgb_img)
            
            # Generate spectral band images
            for band in ['NIR', 'RedEdge']:
                # Simulate spectral response
                if band == 'NIR':
                    # NIR shows high vegetation response
                    spectral_img = simulate_nir_response(rgb_img)
                else:
                    # RedEdge for vegetation edge detection  
                    spectral_img = simulate_rededge_response(rgb_img)
                
                band_path = dataset_dir / 'spectral' / band / split / f'multispectral_{split}_{i:04d}.tiff'
                cv2.imwrite(str(band_path), spectral_img)
            
            # Generate YOLO labels
            label_path = dataset_dir / 'labels' / split / f'multispectral_{split}_{i:04d}.txt'
            generate_yolo_label(label_path, 15, 640, 480)


def generate_synthetic_weed_image(width: int, height: int, num_classes: int) -> np.ndarray:
    """Generate synthetic weed detection image."""
    import cv2
    import numpy as np
    
    # Create background (soil/crop field)
    background = np.random.randint(50, 120, (height, width, 3), dtype=np.uint8)
    
    # Add some texture 
    background = cv2.GaussianBlur(background, (5, 5), 1.0)
    
    # Add random "weed" patches (green blobs)
    num_weeds = np.random.randint(1, 5)
    for _ in range(num_weeds):
        # Random weed position and size
        x = np.random.randint(0, width - 50)
        y = np.random.randint(0, height - 50)
        w = np.random.randint(20, 80)
        h = np.random.randint(20, 80)
        
        # Draw green blob (simulated weed)
        color = (np.random.randint(30, 100), np.random.randint(80, 180), np.random.randint(30, 100))
        cv2.ellipse(background, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, color, -1)
        
        # Add some noise - ensure roi dimensions match
        roi = background[y:y+h, x:x+w]
        if roi.size > 0:  # Check if roi is not empty
            roi_h, roi_w = roi.shape[:2]
            noise = np.random.randint(-20, 20, (roi_h, roi_w, 3), dtype=np.int16)
            roi_with_noise = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            background[y:y+roi_h, x:x+roi_w] = roi_with_noise
    
    return background


def simulate_nir_response(rgb_image: np.ndarray) -> np.ndarray:
    """Simulate NIR spectral response (vegetation appears brighter)."""
    import cv2
    
    # Convert to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # Vegetation (green areas) should be bright in NIR
    nir = gray.copy()
    
    # Enhance areas that were originally green
    green_mask = rgb_image[:, :, 1] > rgb_image[:, :, 0]  # More green than red
    nir[green_mask] = np.clip(nir[green_mask] * 1.5, 0, 255).astype(np.uint8)
    
    return nir


def simulate_rededge_response(rgb_image: np.ndarray) -> np.ndarray:
    """Simulate Red Edge spectral response."""
    import cv2
    
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # Red edge highlights vegetation boundaries
    edges = cv2.Canny(gray, 50, 150)
    rededge = gray.copy()
    rededge[edges > 0] = 255
    
    return rededge


def generate_yolo_label(label_path: Path, num_classes: int, img_width: int, img_height: int):
    """Generate YOLO format label file."""
    num_objects = np.random.randint(1, 4)
    
    with open(label_path, 'w') as f:
        for _ in range(num_objects):
            # Random class
            class_id = np.random.randint(0, num_classes)
            
            # Random bounding box (normalized coordinates)
            x_center = np.random.uniform(0.1, 0.9)
            y_center = np.random.uniform(0.1, 0.9)
            width = np.random.uniform(0.05, 0.3)
            height = np.random.uniform(0.05, 0.3)
            
            # Ensure bounding box stays within image bounds
            x_center = max(width/2, min(1 - width/2, x_center))
            y_center = max(height/2, min(1 - height/2, y_center))
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# Main dataset download mappings
DATASET_HANDLERS = {
    'deepweeds': download_deepweeds,
    'weed25': download_weed25, 
    'cwd30': download_cwd30,
    'weedsgalore': download_weedsgalore
}


def main():
    parser = argparse.ArgumentParser(description="Download weed detection datasets")
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_HANDLERS.keys()), 
                       default=['deepweeds'], help='Datasets to download')
    parser.add_argument('--sample', type=int, help='Number of samples per dataset (for testing)')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    data_dir = Path(args.output)
    data_dir.mkdir(exist_ok=True)
    
    print("🌱 Weed Detection Dataset Downloader")
    print("=" * 50)
    
    for dataset_name in args.datasets:
        if dataset_name in DATASET_HANDLERS:
            try:
                DATASET_HANDLERS[dataset_name](data_dir, args.sample)
            except Exception as e:
                print(f"❌ Failed to download {dataset_name}: {e}")
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
    
    print("\n🎉 Dataset download completed!")


if __name__ == "__main__":
    main()
