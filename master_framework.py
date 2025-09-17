"""
Comprehensive Weed Detection Model Comparison - Master Script
===========================================================

This is the master script that demonstrates the complete comparative study
system for state-of-the-art object detection models in weed detection.

Features implemented:
✅ 9 different model architectures (YOLOv8, YOLOv11, YOLO-NAS, YOLOX, YOLOv7, PP-YOLOE, EfficientDet, DETR, RT-DETR)
✅ 4 specialized weed datasets (Weed25, DeepWeeds, CWD30, WeedsGalore)  
✅ Standardized training protocols with consistent hyperparameters
✅ Comprehensive evaluation framework with detailed metrics
✅ Statistical analysis and significance testing
✅ Advanced visualization and reporting tools
✅ Experiment management and configuration system
✅ Publication-ready figures and LaTeX report generation

This script provides a complete research-ready framework for comparing
object detection models for agricultural weed detection applications.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

# Add scripts to path
sys.path.append('scripts')

# Import our modules
from comprehensive_weed_comparison import ModelComparison
from experiment_manager import ExperimentManager
from advanced_evaluator import AdvancedEvaluator
from advanced_visualizer import AdvancedVisualizer

class WeedDetectionMasterFramework:
    """Master framework for weed detection model comparison studies."""
    
    def __init__(self, base_output_dir: str = "results/master_study"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_comparison = ModelComparison(str(self.base_output_dir / "model_comparison"))
        self.experiment_manager = ExperimentManager("configs/experiments")
        self.evaluator = AdvancedEvaluator(str(self.base_output_dir / "evaluation"))
        self.visualizer = AdvancedVisualizer(str(self.base_output_dir / "visualizations"))
        
        print("🚀 Weed Detection Master Framework Initialized")
        print(f"📁 Base output directory: {self.base_output_dir}")
    
    def run_quick_validation(self) -> Dict[str, str]:
        """Run a quick validation study to test the complete pipeline."""
        
        print("\\n" + "="*60)
        print("🔬 RUNNING QUICK VALIDATION STUDY")
        print("="*60)
        
        # Step 1: Create and run quick experiment
        print("\\n📋 Step 1: Creating quick validation experiment...")
        
        quick_config = self.experiment_manager.create_experiment_config(
            name="quick_validation_demo",
            description="Quick validation of the complete framework",
            model_preset="quick_test",
            dataset_preset="quick_test",
            training_preset="fast",
            epochs=5,  # Very quick for demo
            patience=3
        )
        
        config_path = self.experiment_manager.save_config(quick_config)
        print(f"✅ Configuration saved: {config_path}")
        
        # Step 2: Run model comparison
        print("\\n🏃 Step 2: Running model comparison...")
        
        self.model_comparison.run_comprehensive_comparison(
            models=['yolov8n'],  # Single model for quick demo
            datasets=['dummy'],
            epochs=5,
            patience=3
        )
        
        print("✅ Model comparison completed")
        
        # Step 3: Advanced evaluation (if we have trained models)
        print("\\n📊 Step 3: Advanced evaluation...")
        
        results_file = self.base_output_dir / "model_comparison" / "comparison_results.csv"
        if results_file.exists():
            print(f"✅ Results found: {results_file}")
        else:
            print("⚠️ No results file found, skipping advanced evaluation")
        
        # Step 4: Generate visualizations
        print("\\n🎨 Step 4: Generating visualizations...")
        
        if results_file.exists():
            import pandas as pd
            results_df = pd.read_csv(results_file)
            
            # Create comprehensive visualization package
            viz_files = self.visualizer.create_comprehensive_report(results_df)
            print(f"✅ Visualizations created: {len(viz_files)} file types")
        else:
            print("⚠️ Skipping visualizations due to missing results")
            viz_files = {}
        
        print("\\n✅ Quick validation study completed!")
        
        return {
            'config_path': str(config_path),
            'results_path': str(results_file) if results_file.exists() else None,
            'visualization_files': viz_files
        }
    
    def run_comprehensive_study(self) -> Dict[str, str]:
        """Run a comprehensive research study across multiple models and datasets."""
        
        print("\\n" + "="*60)
        print("🔬 RUNNING COMPREHENSIVE RESEARCH STUDY")
        print("="*60)
        
        # Step 1: Create comprehensive experiment configuration
        print("\\n📋 Step 1: Creating comprehensive experiment...")
        
        comprehensive_config = self.experiment_manager.create_experiment_config(
            name="comprehensive_weed_detection_study",
            description="Complete comparative study of detection models for weed detection",
            model_preset="yolo_all",  # All YOLO models
            dataset_preset="all_real",  # All real datasets
            training_preset="standard",
            epochs=100,
            patience=20,
            augmentation_preset="agriculture_optimized"
        )
        
        config_path = self.experiment_manager.save_config(comprehensive_config)
        print(f"✅ Configuration saved: {config_path}")
        print(self.experiment_manager.generate_experiment_summary(comprehensive_config))
        
        # Step 2: Run comprehensive comparison
        print("\\n🏃 Step 2: Running comprehensive model comparison...")
        print("⚠️ This will take several hours with full configuration")
        
        user_input = input("Continue with comprehensive study? (y/N): ")
        if user_input.lower() not in ['y', 'yes']:
            print("📋 Comprehensive study configuration created but not executed.")
            return {'config_path': str(config_path)}
        
        # Run the full comparison
        self.model_comparison.run_comprehensive_comparison(
            models=comprehensive_config.models,
            datasets=comprehensive_config.datasets,
            epochs=comprehensive_config.epochs,
            patience=comprehensive_config.patience
        )
        
        results_file = self.base_output_dir / "model_comparison" / "comparison_results.csv"
        
        # Step 3: Advanced evaluation and statistical analysis
        print("\\n📊 Step 3: Advanced evaluation and statistical analysis...")
        
        if results_file.exists():
            import pandas as pd
            results_df = pd.read_csv(results_file)
            
            # Perform statistical comparisons
            comparisons = self.evaluator.compare_models_statistical(results_df)
            
            # Generate evaluation report
            eval_report = self.evaluator.generate_evaluation_report()
            print(f"✅ Evaluation report generated")
        
        # Step 4: Comprehensive visualization and reporting
        print("\\n🎨 Step 4: Generating comprehensive reports...")
        
        if results_file.exists():
            viz_files = self.visualizer.create_comprehensive_report(
                results_df, 
                comprehensive_config.__dict__
            )
            print(f"✅ Comprehensive visualization package created")
        
        print("\\n🎉 Comprehensive research study completed!")
        
        return {
            'config_path': str(config_path),
            'results_path': str(results_file) if results_file.exists() else None,
            'visualization_files': viz_files if results_file.exists() else {},
            'evaluation_report': eval_report if results_file.exists() else None
        }
    
    def demonstrate_capabilities(self):
        """Demonstrate the capabilities of the framework."""
        
        print("\\n" + "="*60)
        print("🌟 WEED DETECTION FRAMEWORK CAPABILITIES")
        print("="*60)
        
        print("\\n📋 AVAILABLE MODELS:")
        available_models = self.model_comparison.get_available_models()
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")
        
        print("\\n📁 AVAILABLE DATASETS:")
        datasets = list(self.model_comparison.dataset_configs.keys())
        for i, dataset in enumerate(datasets, 1):
            config = self.model_comparison.dataset_configs[dataset]
            print(f"  {i}. {config.name} ({config.num_classes} classes, {config.train_images} train images)")
        
        print("\\n🔧 FRAMEWORK COMPONENTS:")
        components = [
            "ModelComparison - Unified training and evaluation across multiple architectures",
            "ExperimentManager - Configuration management and experiment orchestration", 
            "AdvancedEvaluator - Detailed metrics, statistical analysis, robustness testing",
            "AdvancedVisualizer - Interactive dashboards, publication figures, LaTeX reports"
        ]
        for i, component in enumerate(components, 1):
            print(f"  {i}. {component}")
        
        print("\\n📊 EVALUATION METRICS:")
        metrics = [
            "Detection Performance: mAP@50, mAP@50-95, precision, recall, F1-score",
            "Speed Metrics: Inference time, FPS, preprocessing/postprocessing time", 
            "Efficiency: Model size, parameter count, memory usage, FLOPS",
            "Robustness: Noise tolerance, blur resistance, brightness adaptation",
            "Statistical: Significance testing, effect sizes, confidence intervals"
        ]
        for i, metric in enumerate(metrics, 1):
            print(f"  {i}. {metric}")
        
        print("\\n🎨 VISUALIZATION OUTPUTS:")
        viz_outputs = [
            "Interactive dashboards with plotly",
            "Statistical analysis plots", 
            "Model architecture comparisons",
            "Dataset difficulty analysis",
            "Publication-ready figures",
            "LaTeX academic reports",
            "HTML summary pages"
        ]
        for i, output in enumerate(viz_outputs, 1):
            print(f"  {i}. {output}")
        
        print("\\n⚙️ CONFIGURATION PRESETS:")
        presets = [
            "Model presets: quick_test, yolo_all, comprehensive, etc.",
            "Dataset presets: agricultural, rangeland, aerial, comprehensive",
            "Training presets: fast, standard, thorough, research",
            "Augmentation presets: light, medium, heavy, agriculture_optimized"
        ]
        for i, preset in enumerate(presets, 1):
            print(f"  {i}. {preset}")
    
    def show_usage_examples(self):
        """Show usage examples for different scenarios."""
        
        print("\\n" + "="*60)
        print("💡 USAGE EXAMPLES")
        print("="*60)
        
        examples = [
            {
                "title": "Quick Model Testing",
                "description": "Test a few models quickly on dummy data",
                "command": "python master_framework.py --quick-validation"
            },
            {
                "title": "YOLO Family Comparison", 
                "description": "Compare all YOLO variants on real datasets",
                "command": "python scripts/comprehensive_weed_comparison.py --models yolov8n yolov8s yolov11n yolov11s --datasets weed25 deepweeds --epochs 50"
            },
            {
                "title": "Full Research Study",
                "description": "Complete comparative study for publication",
                "command": "python master_framework.py --comprehensive-study"
            },
            {
                "title": "Custom Experiment",
                "description": "Create custom experiment configuration",
                "command": "python scripts/experiment_manager.py --create-custom my_experiment --models yolov8m yolov11m --epochs 100"
            },
            {
                "title": "Visualization Only",
                "description": "Generate visualizations from existing results",
                "command": "python scripts/advanced_visualizer.py --results results/comparison_results.csv"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\\n{i}. {example['title']}")
            print(f"   {example['description']}")
            print(f"   Command: {example['command']}")

def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Weed Detection Model Comparison Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_framework.py --quick-validation
  python master_framework.py --comprehensive-study  
  python master_framework.py --demo
  python master_framework.py --examples
        """
    )
    
    parser.add_argument('--quick-validation', action='store_true',
                       help='Run quick validation study')
    parser.add_argument('--comprehensive-study', action='store_true',
                       help='Run comprehensive research study')
    parser.add_argument('--demo', action='store_true',
                       help='Demonstrate framework capabilities')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples')
    parser.add_argument('--output-dir', type=str, default='results/master_study',
                       help='Base output directory')
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = WeedDetectionMasterFramework(args.output_dir)
    
    if args.quick_validation:
        results = framework.run_quick_validation()
        print("\\n📋 Quick Validation Results:")
        for key, value in results.items():
            if value:
                print(f"  {key}: {value}")
    
    elif args.comprehensive_study:
        results = framework.run_comprehensive_study()
        print("\\n📋 Comprehensive Study Results:")
        for key, value in results.items():
            if value:
                print(f"  {key}: {value}")
    
    elif args.demo:
        framework.demonstrate_capabilities()
    
    elif args.examples:
        framework.show_usage_examples()
    
    else:
        print("\\n🌟 Comprehensive Weed Detection Model Comparison Framework")
        print("="*60)
        print("\\nThis framework provides a complete research environment for comparing")
        print("state-of-the-art object detection models on agricultural weed detection tasks.")
        print("\\nUse --help to see available options or:")
        print("  --demo for capability overview")
        print("  --examples for usage examples") 
        print("  --quick-validation for a quick test")
        print("  --comprehensive-study for full research study")
        
        # Show brief capabilities
        framework.demonstrate_capabilities()

if __name__ == "__main__":
    main()