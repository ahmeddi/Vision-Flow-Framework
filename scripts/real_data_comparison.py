#!/usr/bin/env python3
"""
Real Data Weed Detection Comparison
===================================

This script runs comprehensive comparisons using the actual weed datasets
available in the project. It focuses on practical experiments that can
complete in reasonable time while providing meaningful results.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add scripts to path
sys.path.append('scripts')

from comprehensive_weed_comparison import ModelComparison
from advanced_visualizer import AdvancedVisualizer

def setup_logging():
    """Set up logging for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('real_data_comparison.log'),
            logging.StreamHandler()
        ]
    )

def run_quick_real_comparison():
    """Run a quick comparison on real data for validation."""
    
    print("🔬 Running Quick Real Data Comparison")
    print("=" * 50)
    
    # Initialize comparison
    comparison = ModelComparison("results/real_data_quick")
    
    # Quick configuration - fastest models on most manageable dataset
    models = ['yolov8n']  # Start with fastest model
    datasets = ['deepweeds']  # 8 classes, good size
    epochs = 20
    patience = 8
    
    print(f"📋 Configuration:")
    print(f"   Models: {models}")
    print(f"   Datasets: {datasets}")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience}")
    
    # Run comparison
    comparison.run_comprehensive_comparison(
        models=models,
        datasets=datasets,
        epochs=epochs,
        patience=patience
    )
    
    print("✅ Quick comparison completed!")
    return "results/real_data_quick/comparison_results.csv"

def run_multi_model_comparison():
    """Run comparison across multiple models on real data."""
    
    print("🔬 Running Multi-Model Real Data Comparison")
    print("=" * 50)
    
    # Initialize comparison
    comparison = ModelComparison("results/real_data_multi")
    
    # Multi-model configuration
    models = ['yolov8n', 'yolov8s', 'yolov11n']  # Different model sizes
    datasets = ['deepweeds', 'weed25']  # Two different datasets
    epochs = 25
    patience = 8
    
    print(f"📋 Configuration:")
    print(f"   Models: {models}")
    print(f"   Datasets: {datasets}")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience}")
    print(f"   Total experiments: {len(models) * len(datasets)}")
    
    # Estimate time
    estimated_time = len(models) * len(datasets) * 15  # ~15 min per experiment
    print(f"   Estimated time: ~{estimated_time} minutes")
    
    # Ask for confirmation
    user_input = input("\nContinue with multi-model comparison? (y/N): ")
    if user_input.lower() not in ['y', 'yes']:
        print("❌ Multi-model comparison cancelled.")
        return None
    
    # Run comparison
    comparison.run_comprehensive_comparison(
        models=models,
        datasets=datasets,
        epochs=epochs,
        patience=patience
    )
    
    print("✅ Multi-model comparison completed!")
    return "results/real_data_multi/comparison_results.csv"

def run_comprehensive_real_study():
    """Run comprehensive study across all real datasets."""
    
    print("🔬 Running Comprehensive Real Data Study")
    print("=" * 50)
    
    # Initialize comparison
    comparison = ModelComparison("results/real_data_comprehensive")
    
    # Comprehensive configuration
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov11n', 'yolov11s']
    datasets = ['deepweeds', 'weed25', 'cwd30', 'weedsgalore']
    epochs = 50
    patience = 15
    
    print(f"📋 Configuration:")
    print(f"   Models: {models}")
    print(f"   Datasets: {datasets}")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience}")
    print(f"   Total experiments: {len(models) * len(datasets)}")
    
    # Estimate time
    estimated_time = len(models) * len(datasets) * 25  # ~25 min per experiment
    estimated_hours = estimated_time / 60
    print(f"   Estimated time: ~{estimated_time} minutes ({estimated_hours:.1f} hours)")
    
    # Ask for confirmation
    print("\n⚠️  This is a long-running comprehensive study!")
    user_input = input("Continue with comprehensive study? (y/N): ")
    if user_input.lower() not in ['y', 'yes']:
        print("❌ Comprehensive study cancelled.")
        return None
    
    # Run comparison
    comparison.run_comprehensive_comparison(
        models=models,
        datasets=datasets,
        epochs=epochs,
        patience=patience
    )
    
    print("✅ Comprehensive study completed!")
    return "results/real_data_comprehensive/comparison_results.csv"

def generate_analysis(results_file):
    """Generate analysis and visualizations from results."""
    
    if not results_file or not os.path.exists(results_file):
        print("❌ No results file found for analysis")
        return
    
    print(f"\n📊 Generating Analysis from: {results_file}")
    print("=" * 50)
    
    # Initialize visualizer
    output_dir = Path(results_file).parent / "analysis"
    visualizer = AdvancedVisualizer(str(output_dir))
    
    # Load results
    import pandas as pd
    results_df = pd.read_csv(results_file)
    
    print(f"📈 Results loaded: {len(results_df)} experiments")
    print(f"   Models: {results_df['model_name'].unique().tolist()}")
    print(f"   Datasets: {results_df['dataset_name'].unique().tolist()}")
    
    # Generate comprehensive report
    viz_files = visualizer.create_comprehensive_report(results_df)
    
    print("✅ Analysis completed!")
    print(f"📁 Output directory: {output_dir}")
    
    for report_type, files in viz_files.items():
        if isinstance(files, list):
            print(f"   {report_type}: {len(files)} files")
        else:
            print(f"   {report_type}: {Path(files).name}")

def main():
    """Main function with command-line interface."""
    
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Real Data Weed Detection Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python real_data_comparison.py --quick
  python real_data_comparison.py --multi-model
  python real_data_comparison.py --comprehensive
  python real_data_comparison.py --analyze results/real_data_quick/comparison_results.csv
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation on real data')
    parser.add_argument('--multi-model', action='store_true',
                       help='Run multi-model comparison')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive study on all real datasets')
    parser.add_argument('--analyze', type=str,
                       help='Generate analysis from existing results file')
    
    args = parser.parse_args()
    
    if args.quick:
        results_file = run_quick_real_comparison()
        generate_analysis(results_file)
    
    elif args.multi_model:
        results_file = run_multi_model_comparison()
        if results_file:
            generate_analysis(results_file)
    
    elif args.comprehensive:
        results_file = run_comprehensive_real_study()
        if results_file:
            generate_analysis(results_file)
    
    elif args.analyze:
        generate_analysis(args.analyze)
    
    else:
        print("\n🌾 Real Data Weed Detection Comparison")
        print("=" * 50)
        print("\nThis script provides different levels of comparison studies using real weed datasets:")
        print("  --quick: Fast validation (YOLOv8n on DeepWeeds, ~20 minutes)")
        print("  --multi-model: Multiple models comparison (~1.5 hours)")
        print("  --comprehensive: Full study on all datasets (~8+ hours)")
        print("  --analyze: Generate visualizations from existing results")
        print("\nAll real datasets available:")
        
        # Show available datasets
        try:
            from comprehensive_weed_comparison import ModelComparison
            mc = ModelComparison()
            for name, config in mc.dataset_configs.items():
                if name != 'dummy':
                    print(f"  📁 {config.name}: {config.num_classes} classes, {config.train_images} train images")
        except Exception as e:
            print(f"  ❌ Error loading dataset info: {e}")
        
        print("\nUse --help for detailed options.")

if __name__ == "__main__":
    main()