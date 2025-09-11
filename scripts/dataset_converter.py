"""Dataset conversion utilities for different annotation formats."""
import json
import csv
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Any

class DatasetConverter:
    """Convert various annotation formats to YOLO format."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def deepweeds_to_yolo(self, 
                         images_dir: str, 
                         labels_csv: str,
                         train_ratio: float = 0.8) -> Dict[str, Any]:
        """Convert DeepWeeds CSV format to YOLO format."""
        
        images_path = Path(images_dir)
        
        # Read CSV labels
        df = pd.read_csv(labels_csv)
        
        # Class mapping
        class_names = sorted(df['Species'].unique().tolist())
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        # Create output directories
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Split data
        train_samples = df.sample(frac=train_ratio, random_state=42)
        val_samples = df.drop(train_samples.index)
        
        def process_split(samples: pd.DataFrame, split: str):
            for _, row in samples.iterrows():
                filename = row['Filename']
                species = row['Species']
                
                # Copy image
                src_img = images_path / filename
                dst_img = self.output_dir / 'images' / split / filename
                
                if src_img.exists():
                    # Copy image file
                    import shutil
                    shutil.copy2(src_img, dst_img)
                    
                    # Create YOLO label (assuming full image annotation)
                    img = Image.open(src_img)
                    w, h = img.size
                    
                    # Full image bounding box (center format)
                    label_content = f"{class_to_id[species]} 0.5 0.5 1.0 1.0\n"
                    
                    # Save label
                    label_file = self.output_dir / 'labels' / split / f"{Path(filename).stem}.txt"
                    label_file.write_text(label_content)
        
        # Process both splits
        process_split(train_samples, 'train')
        process_split(val_samples, 'val')
        
        # Create data.yaml
        data_yaml = {
            'train': str(self.output_dir / 'images' / 'train'),
            'val': str(self.output_dir / 'images' / 'val'),
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        
        return {
            'num_classes': len(class_names),
            'class_names': class_names,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'data_yaml': str(self.output_dir / 'data.yaml')
        }
    
    def coco_to_yolo(self, 
                     images_dir: str, 
                     annotations_file: str) -> Dict[str, Any]:
        """Convert COCO format annotations to YOLO format."""
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Extract information
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat for cat in coco_data['categories']}
        
        # Create class mapping
        class_names = [categories[cat_id]['name'] for cat_id in sorted(categories.keys())]
        cat_id_to_yolo_id = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
        
        # Create output directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process annotations
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Convert each image
        for img_id, img_info in images.items():
            filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img = Path(images_dir) / filename
            dst_img = self.output_dir / 'images' / filename
            
            if src_img.exists():
                import shutil
                shutil.copy2(src_img, dst_img)
                
                # Convert annotations
                yolo_annotations = []
                if img_id in image_annotations:
                    for ann in image_annotations[img_id]:
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        
                        # Normalize coordinates
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        # Get class ID
                        class_id = cat_id_to_yolo_id[ann['category_id']]
                        
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save label file
                label_file = self.output_dir / 'labels' / f"{Path(filename).stem}.txt"
                label_file.write_text('\n'.join(yolo_annotations))
        
        # Create data.yaml
        data_yaml = {
            'train': str(self.output_dir / 'images'),
            'val': str(self.output_dir / 'images'),  # Same for now, split later
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        
        return {
            'num_classes': len(class_names),
            'class_names': class_names,
            'num_images': len(images),
            'data_yaml': str(self.output_dir / 'data.yaml')
        }
    
    def xml_to_yolo(self, 
                    images_dir: str, 
                    annotations_dir: str) -> Dict[str, Any]:
        """Convert XML (Pascal VOC) format annotations to YOLO format."""
        
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)
        
        # Collect all classes
        all_classes = set()
        xml_files = list(annotations_path.glob('*.xml'))
        
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                all_classes.add(class_name)
        
        # Create class mapping
        class_names = sorted(list(all_classes))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        # Create output directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process each annotation
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Copy image
            src_img = images_path / filename
            dst_img = self.output_dir / 'images' / filename
            
            if src_img.exists():
                import shutil
                shutil.copy2(src_img, dst_img)
                
                # Convert annotations
                yolo_annotations = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    x_center = (xmin + xmax) / (2 * img_width)
                    y_center = (ymin + ymax) / (2 * img_height)
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    class_id = class_to_id[class_name]
                    
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save label file
                label_file = self.output_dir / 'labels' / f"{Path(filename).stem}.txt"
                if yolo_annotations:
                    label_file.write_text('\n'.join(yolo_annotations))
                else:
                    label_file.write_text('')  # Empty file for images without annotations
        
        # Create data.yaml
        data_yaml = {
            'train': str(self.output_dir / 'images'),
            'val': str(self.output_dir / 'images'),  # Same for now, split later
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        
        return {
            'num_classes': len(class_names),
            'class_names': class_names,
            'num_images': len(xml_files),
            'data_yaml': str(self.output_dir / 'data.yaml')
        }
    
    def split_dataset(self, 
                     data_yaml_path: str,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.05) -> Dict[str, Any]:
        """Split dataset into train/val/test sets."""
        
        # Read current data.yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get all images
        images_dir = Path(data_config['train'])
        all_images = list(images_dir.glob('*.[jp][pn]g'))
        
        # Split images
        import random
        random.shuffle(all_images)
        
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train+n_val]
        test_images = all_images[n_train+n_val:]
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        def move_files(image_list: List[Path], split: str):
            for img_path in image_list:
                # Move image
                dst_img = self.output_dir / 'images' / split / img_path.name
                img_path.rename(dst_img)
                
                # Move label
                label_path = images_dir.parent / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    dst_label = self.output_dir / 'labels' / split / f"{img_path.stem}.txt"
                    label_path.rename(dst_label)
        
        # Move files to appropriate splits
        move_files(train_images, 'train')
        move_files(val_images, 'val')
        move_files(test_images, 'test')
        
        # Update data.yaml
        data_config.update({
            'train': str(self.output_dir / 'images' / 'train'),
            'val': str(self.output_dir / 'images' / 'val'),
            'test': str(self.output_dir / 'images' / 'test') if test_images else None
        })
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f)
        
        return {
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'test_samples': len(test_images),
            'data_yaml': str(self.output_dir / 'data.yaml')
        }
