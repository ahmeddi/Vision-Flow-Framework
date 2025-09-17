"""
Experiment Configuration Manager for Weed Detection Model Comparison
================================================================

This module provides configuration management for running comprehensive
experiments across multiple models and datasets with standardized protocols.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment run."""
    
    # Experiment metadata
    experiment_name: str
    description: str
    author: str
    timestamp: str
    
    # Model configuration
    models: List[str]
    exclude_models: List[str]
    model_families: List[str]  # e.g., ['yolov8', 'yolov11', 'yolo_nas']
    
    # Dataset configuration
    datasets: List[str]
    exclude_datasets: List[str]
    
    # Training configuration
    epochs: int
    patience: int
    batch_size: int
    image_size: int
    device: str
    
    # Optimization settings
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float
    
    # Data augmentation
    augmentation_preset: str  # 'light', 'medium', 'heavy', 'custom'
    custom_augmentation: Dict[str, Any]
    
    # Evaluation settings
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    
    # Output configuration
    output_dir: str
    save_models: bool
    save_predictions: bool
    generate_plots: bool
    generate_reports: bool
    
    # Resource constraints
    max_training_time_hours: float
    max_memory_gb: float
    parallel_experiments: int

@dataclass
class DatasetPreprocessingConfig:
    """Configuration for dataset preprocessing."""
    
    resize_method: str  # 'stretch', 'pad', 'crop'
    normalize: bool
    mean: List[float]
    std: List[float]
    train_val_test_split: List[float]  # [0.7, 0.2, 0.1]
    min_object_size: int
    max_objects_per_image: int

class ExperimentManager:
    """Manager for creating and running experiment configurations."""
    
    def __init__(self, config_dir: str = "configs/experiments"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load predefined configurations
        self.model_presets = self._load_model_presets()
        self.dataset_presets = self._load_dataset_presets()
        self.augmentation_presets = self._load_augmentation_presets()
    
    def _load_model_presets(self) -> Dict[str, List[str]]:
        """Load predefined model configurations."""
        return {
            'yolo_small': ['yolov8n', 'yolov8s', 'yolov11n', 'yolov11s'],
            'yolo_medium': ['yolov8m', 'yolov8l', 'yolov11m', 'yolov11l'],
            'yolo_large': ['yolov8x', 'yolov11x'],
            'yolo_all': [
                'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'
            ],
            'all_architectures': [
                'yolov8n', 'yolov8s', 'yolov8m', 'yolov11n', 'yolov11s',
                'yolo_nas_s', 'yolo_nas_m', 'yolox_s', 'yolox_m',
                'efficientdet_d0', 'efficientdet_d1', 'detr_resnet50'
            ],
            'quick_test': ['yolov8n', 'yolov11n'],
            'comprehensive': [
                # YOLOv8 family
                'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                # YOLOv11 family  
                'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x',
                # YOLO-NAS
                'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l',
                # YOLOX
                'yolox_nano', 'yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l',
                # EfficientDet
                'efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2',
                # Transformers
                'detr_resnet50', 'rtdetr_resnet50'
            ]
        }
    
    def _load_dataset_presets(self) -> Dict[str, List[str]]:
        """Load predefined dataset configurations."""
        return {
            'all_real': ['weed25', 'deepweeds', 'cwd30', 'weedsgalore'],
            'agricultural': ['weed25', 'cwd30'],
            'rangeland': ['deepweeds'],
            'aerial': ['weedsgalore'],
            'quick_test': ['dummy'],
            'diverse': ['weed25', 'deepweeds', 'weedsgalore'],
            'comprehensive': ['weed25', 'deepweeds', 'cwd30', 'weedsgalore']
        }
    
    def _load_augmentation_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined augmentation configurations."""
        return {
            'light': {
                'hsv_h': 0.01,
                'hsv_s': 0.5,
                'hsv_v': 0.3,
                'degrees': 5.0,
                'translate': 0.05,
                'scale': 0.2,
                'fliplr': 0.5,
                'mosaic': 0.5,
                'mixup': 0.0
            },
            'medium': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.1
            },
            'heavy': {
                'hsv_h': 0.02,
                'hsv_s': 0.9,
                'hsv_v': 0.5,
                'degrees': 15.0,
                'translate': 0.15,
                'scale': 0.7,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.2,
                'copy_paste': 0.1
            },
            'agriculture_optimized': {
                'hsv_h': 0.01,  # Preserve natural colors
                'hsv_s': 0.6,
                'hsv_v': 0.4,
                'degrees': 8.0,  # Realistic rotation
                'translate': 0.08,
                'scale': 0.3,
                'fliplr': 0.5,
                'mosaic': 0.8,
                'mixup': 0.05,
                'perspective': 0.0001,  # Slight perspective changes
                'copy_paste': 0.05
            }
        }
    
    def create_experiment_config(self, 
                               name: str,
                               description: str = "",
                               model_preset: str = "quick_test",
                               dataset_preset: str = "quick_test",
                               training_preset: str = "fast",
                               custom_models: List[str] = None,
                               custom_datasets: List[str] = None,
                               **kwargs) -> ExperimentConfig:
        """Create a new experiment configuration."""
        
        # Resolve model and dataset presets
        if custom_models:
            models = custom_models
        else:
            models = self.model_presets.get(model_preset, ['yolov8n'])
        
        if custom_datasets:
            datasets = custom_datasets
        else:
            datasets = self.dataset_presets.get(dataset_preset, ['dummy'])
        
        # Training presets
        training_configs = {
            'fast': {
                'epochs': 10,
                'patience': 5,
                'batch_size': 16,
                'learning_rate': 0.01
            },
            'standard': {
                'epochs': 100,
                'patience': 20,
                'batch_size': 16,
                'learning_rate': 0.01
            },
            'thorough': {
                'epochs': 200,
                'patience': 30,
                'batch_size': 16,
                'learning_rate': 0.01
            },
            'research': {
                'epochs': 300,
                'patience': 50,
                'batch_size': 32,
                'learning_rate': 0.008
            }
        }
        
        training_config = training_configs.get(training_preset, training_configs['fast'])
        
        # Create configuration
        config = ExperimentConfig(
            experiment_name=name,
            description=description,
            author=os.getenv('USER', 'researcher'),
            timestamp=datetime.now().isoformat(),
            
            # Models and datasets
            models=models,
            exclude_models=[],
            model_families=[],
            datasets=datasets,
            exclude_datasets=[],
            
            # Training settings
            epochs=training_config['epochs'],
            patience=training_config['patience'],
            batch_size=training_config['batch_size'],
            image_size=640,
            device='auto',
            
            # Optimization
            optimizer='auto',
            learning_rate=training_config['learning_rate'],
            weight_decay=0.0005,
            momentum=0.937,
            
            # Augmentation
            augmentation_preset='medium',
            custom_augmentation={},
            
            # Evaluation
            confidence_threshold=0.25,
            iou_threshold=0.7,
            max_detections=300,
            
            # Output
            output_dir=f"results/experiments/{name}",
            save_models=True,
            save_predictions=True,
            generate_plots=True,
            generate_reports=True,
            
            # Resources
            max_training_time_hours=24.0,
            max_memory_gb=16.0,
            parallel_experiments=1
        )
        
        # Apply any custom overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def save_config(self, config: ExperimentConfig, filename: str = None) -> Path:
        """Save experiment configuration to file."""
        
        if filename is None:
            filename = f"{config.experiment_name}_{config.timestamp[:10]}.yaml"
        
        config_path = self.config_dir / filename
        
        # Convert to dictionary and save
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {config_path}")
        return config_path
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """Load experiment configuration from file."""
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ExperimentConfig(**config_dict)
    
    def create_predefined_experiments(self) -> List[Path]:
        """Create a set of predefined experiment configurations."""
        
        experiments = []
        
        # 1. Quick validation experiment
        quick_config = self.create_experiment_config(
            name="quick_validation",
            description="Quick validation test with minimal models and datasets",
            model_preset="quick_test",
            dataset_preset="quick_test",
            training_preset="fast"
        )
        experiments.append(self.save_config(quick_config))
        
        # 2. YOLO family comparison
        yolo_config = self.create_experiment_config(
            name="yolo_family_comparison",
            description="Comprehensive comparison of YOLOv8 and YOLOv11 models",
            model_preset="yolo_all",
            dataset_preset="all_real",
            training_preset="standard"
        )
        experiments.append(self.save_config(yolo_config))
        
        # 3. Architecture comparison
        arch_config = self.create_experiment_config(
            name="architecture_comparison",
            description="Compare different detection architectures (YOLO, EfficientDet, DETR)",
            model_preset="all_architectures",
            dataset_preset="diverse",
            training_preset="thorough",
            epochs=150,
            augmentation_preset="agriculture_optimized"
        )
        experiments.append(self.save_config(arch_config))
        
        # 4. Agricultural optimization study
        agri_config = self.create_experiment_config(
            name="agricultural_optimization",
            description="Optimized study for agricultural weed detection",
            model_preset="yolo_medium",
            dataset_preset="agricultural",
            training_preset="research",
            augmentation_preset="agriculture_optimized",
            confidence_threshold=0.3,
            epochs=250,
            patience=40
        )
        experiments.append(self.save_config(agri_config))
        
        # 5. Comprehensive research study
        comprehensive_config = self.create_experiment_config(
            name="comprehensive_research_study",
            description="Complete comparative study for research publication",
            model_preset="comprehensive",
            dataset_preset="comprehensive",
            training_preset="research",
            epochs=300,
            patience=50,
            batch_size=32,
            augmentation_preset="agriculture_optimized",
            max_training_time_hours=72.0
        )
        experiments.append(self.save_config(comprehensive_config))
        
        print(f"Created {len(experiments)} predefined experiment configurations")
        return experiments
    
    def generate_experiment_summary(self, config: ExperimentConfig) -> str:
        """Generate a human-readable summary of the experiment."""
        
        summary = []
        summary.append(f"# Experiment: {config.experiment_name}")
        summary.append(f"**Description:** {config.description}")
        summary.append(f"**Author:** {config.author}")
        summary.append(f"**Date:** {config.timestamp[:10]}")
        summary.append("")
        
        # Models
        summary.append("## Models")
        summary.append(f"- **Count:** {len(config.models)}")
        summary.append(f"- **Models:** {', '.join(config.models[:5])}")
        if len(config.models) > 5:
            summary.append(f"  ... and {len(config.models) - 5} more")
        summary.append("")
        
        # Datasets
        summary.append("## Datasets")
        summary.append(f"- **Count:** {len(config.datasets)}")
        summary.append(f"- **Datasets:** {', '.join(config.datasets)}")
        summary.append("")
        
        # Training
        summary.append("## Training Configuration")
        summary.append(f"- **Epochs:** {config.epochs}")
        summary.append(f"- **Patience:** {config.patience}")
        summary.append(f"- **Batch Size:** {config.batch_size}")
        summary.append(f"- **Learning Rate:** {config.learning_rate}")
        summary.append(f"- **Augmentation:** {config.augmentation_preset}")
        summary.append("")
        
        # Estimated resources
        total_experiments = len(config.models) * len(config.datasets)
        estimated_hours = total_experiments * (config.epochs / 100) * 2  # Rough estimate
        
        summary.append("## Estimated Resources")
        summary.append(f"- **Total Experiments:** {total_experiments}")
        summary.append(f"- **Estimated Time:** {estimated_hours:.1f} hours")
        summary.append(f"- **Output Directory:** {config.output_dir}")
        
        return "\n".join(summary)
    
    def run_experiment_from_config(self, config_path: str):
        """Run an experiment from a configuration file."""
        
        config = self.load_config(config_path)
        
        # Import the comparison framework
        import sys
        sys.path.append('scripts')
        from comprehensive_weed_comparison import ModelComparison
        
        # Create comparison instance
        comparison = ModelComparison(config.output_dir)
        
        print(f"Running experiment: {config.experiment_name}")
        print(self.generate_experiment_summary(config))
        
        # Run the comparison
        comparison.run_comprehensive_comparison(
            models=config.models,
            datasets=config.datasets,
            epochs=config.epochs,
            patience=config.patience
        )
        
        print(f"Experiment completed! Results saved to: {config.output_dir}")

def main():
    """Main function for command-line usage."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Configuration Manager")
    parser.add_argument('--create-presets', action='store_true',
                       help='Create predefined experiment configurations')
    parser.add_argument('--run-config', type=str,
                       help='Run experiment from configuration file')
    parser.add_argument('--summary', type=str,
                       help='Generate summary for configuration file')
    parser.add_argument('--create-custom', type=str,
                       help='Create custom experiment configuration')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Custom models list')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Custom datasets list')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.create_presets:
        manager.create_predefined_experiments()
    
    elif args.run_config:
        manager.run_experiment_from_config(args.run_config)
    
    elif args.summary:
        config = manager.load_config(args.summary)
        print(manager.generate_experiment_summary(config))
    
    elif args.create_custom:
        config = manager.create_experiment_config(
            name=args.create_custom,
            custom_models=args.models,
            custom_datasets=args.datasets,
            epochs=args.epochs
        )
        manager.save_config(config)
        print(manager.generate_experiment_summary(config))
    
    else:
        print("Use --help for available options")

if __name__ == "__main__":
    main()