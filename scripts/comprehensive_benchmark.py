"""Comprehensive benchmarking script for all models and datasets."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import subprocess
import sys

def run_training_benchmark(datasets: List[str], 
                          models: List[str], 
                          output_dir: str = "results/benchmark") -> Dict[str, Any]:
    """Run training benchmark across multiple datasets and models."""
    
    results = {}
    total_combinations = len(datasets) * len(models)
    current = 0
    
    print(f"Starting comprehensive benchmark: {len(datasets)} datasets × {len(models)} models = {total_combinations} combinations")
    
    for dataset in datasets:
        dataset_results = {}
        data_yaml = f"data/{dataset}/data.yaml"
        
        # Check if dataset exists
        if not Path(data_yaml).exists():
            print(f"❌ Dataset {dataset} not found at {data_yaml}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        for model in models:
            current += 1
            print(f"[{current}/{total_combinations}] Training {model} on {dataset}...")
            
            start_time = time.time()
            
            # Run training
            cmd = [
                sys.executable, 
                "scripts/train.py",
                "--data", data_yaml,
                "--models", model,
                "--epochs", "50",  # Reduced for benchmarking
                "--batch-size", "16",
                "--output", f"{output_dir}/training/{dataset}"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
                training_time = time.time() - start_time
                
                if result.returncode == 0:
                    # Parse training results
                    print(f"  ✅ Training completed in {training_time:.1f}s")
                    
                    # Run evaluation
                    eval_cmd = [
                        sys.executable,
                        "scripts/evaluate.py", 
                        "--models_dir", f"{output_dir}/training/{dataset}",
                        "--data", data_yaml
                    ]
                    
                    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)
                    
                    if eval_result.returncode == 0:
                        # Load evaluation results
                        eval_file = Path(f"results/eval_summary.json")
                        if eval_file.exists():
                            with open(eval_file, 'r') as f:
                                eval_data = json.load(f)
                            
                            dataset_results[model] = {
                                'status': 'success',
                                'training_time': training_time,
                                'evaluation': eval_data[-1] if eval_data else {}  # Latest result
                            }
                        else:
                            dataset_results[model] = {
                                'status': 'eval_failed',
                                'training_time': training_time,
                                'error': 'Evaluation results not found'
                            }
                    else:
                        dataset_results[model] = {
                            'status': 'eval_failed', 
                            'training_time': training_time,
                            'error': eval_result.stderr
                        }
                        
                else:
                    print(f"  ❌ Training failed: {result.stderr}")
                    dataset_results[model] = {
                        'status': 'training_failed',
                        'training_time': training_time,
                        'error': result.stderr
                    }
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Training timeout after 2 hours")
                dataset_results[model] = {
                    'status': 'timeout',
                    'training_time': 7200,
                    'error': 'Training timeout'
                }
            except Exception as e:
                print(f"  ❌ Error: {e}")
                dataset_results[model] = {
                    'status': 'error',
                    'training_time': time.time() - start_time,
                    'error': str(e)
                }
        
        results[dataset] = dataset_results
    
    return results

def run_energy_benchmark(models_dir: str, 
                        data_yaml: str,
                        output_file: str = "results/energy_benchmark.json") -> Dict[str, Any]:
    """Run energy consumption benchmark."""
    
    print("Running energy consumption benchmark...")
    
    cmd = [
        sys.executable,
        "scripts/energy_logger.py",
        "--models_dir", models_dir,
        "--data", data_yaml,
        "--n_images", "100", 
        "--output", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            # Load energy results
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    return json.load(f)
            else:
                return {"error": "Energy results file not found"}
        else:
            return {"error": result.stderr}
            
    except Exception as e:
        return {"error": str(e)}

def run_robustness_benchmark(models_dir: str,
                           data_yaml: str, 
                           output_dir: str = "results/robustness") -> Dict[str, Any]:
    """Run robustness evaluation benchmark."""
    
    print("Running robustness benchmark...")
    
    # Find all model weights
    models_path = Path(models_dir)
    weight_files = list(models_path.rglob("best.pt")) + list(models_path.rglob("best.pth"))
    
    results = {}
    
    for weights in weight_files:
        model_name = weights.parent.parent.name  # Extract model name from path
        print(f"  Testing robustness of {model_name}...")
        
        cmd = [
            sys.executable,
            "scripts/perturb_eval.py",
            "--model", str(weights),
            "--data", data_yaml,
            "--output", f"{output_dir}/{model_name}_robustness.json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                # Load robustness results
                result_file = Path(f"{output_dir}/{model_name}_robustness.json")
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        results[model_name] = json.load(f)
                else:
                    results[model_name] = {"error": "Results file not found"}
            else:
                results[model_name] = {"error": result.stderr}
                
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return results

def generate_summary_report(benchmark_results: Dict[str, Any], 
                          output_file: str = "results/benchmark_summary.json"):
    """Generate comprehensive benchmark summary report."""
    
    print("Generating summary report...")
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_experiments': 0,
        'successful_experiments': 0,
        'datasets': list(benchmark_results.keys()),
        'models_tested': [],
        'performance_summary': {},
        'best_performers': {}
    }
    
    # Collect all models tested
    all_models = set()
    for dataset_results in benchmark_results.values():
        all_models.update(dataset_results.keys())
    summary['models_tested'] = sorted(list(all_models))
    
    # Calculate totals and performance metrics
    performance_data = []
    
    for dataset, dataset_results in benchmark_results.items():
        for model, model_results in dataset_results.items():
            summary['total_experiments'] += 1
            
            if model_results['status'] == 'success':
                summary['successful_experiments'] += 1
                
                eval_data = model_results.get('evaluation', {})
                performance_data.append({
                    'dataset': dataset,
                    'model': model,
                    'map50': eval_data.get('map50', 0),
                    'map50_95': eval_data.get('map50_95', 0),
                    'fps': eval_data.get('fps', 0),
                    'latency_ms': eval_data.get('latency_ms', 0),
                    'parameters': eval_data.get('total_parameters', 0),
                    'model_size_mb': eval_data.get('model_size_mb', 0),
                    'training_time': model_results.get('training_time', 0)
                })
    
    # Create performance DataFrame for analysis
    if performance_data:
        df = pd.DataFrame(performance_data)
        
        # Best performers by metric
        summary['best_performers'] = {
            'highest_map50': df.loc[df['map50'].idxmax()].to_dict() if not df.empty else {},
            'highest_fps': df.loc[df['fps'].idxmax()].to_dict() if not df.empty else {},
            'lowest_latency': df.loc[df['latency_ms'].idxmin()].to_dict() if not df.empty else {},
            'smallest_model': df.loc[df['model_size_mb'].idxmin()].to_dict() if not df.empty else {},
            'fastest_training': df.loc[df['training_time'].idxmin()].to_dict() if not df.empty else {}
        }
        
        # Performance summary statistics
        summary['performance_summary'] = {
            'map50_stats': {
                'mean': float(df['map50'].mean()),
                'std': float(df['map50'].std()),
                'min': float(df['map50'].min()),
                'max': float(df['map50'].max())
            },
            'fps_stats': {
                'mean': float(df['fps'].mean()),
                'std': float(df['fps'].std()),
                'min': float(df['fps'].min()),
                'max': float(df['fps'].max())
            },
            'model_size_stats': {
                'mean_mb': float(df['model_size_mb'].mean()),
                'std_mb': float(df['model_size_mb'].std()),
                'min_mb': float(df['model_size_mb'].min()),
                'max_mb': float(df['model_size_mb'].max())
            }
        }
    
    summary['raw_results'] = benchmark_results
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Summary report saved to {output_file}")
    return summary

def print_benchmark_summary(summary: Dict[str, Any]):
    """Print formatted benchmark summary."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Success rate: {summary['successful_experiments']/summary['total_experiments']*100:.1f}%")
    
    print(f"\nDatasets tested: {len(summary['datasets'])}")
    for dataset in summary['datasets']:
        print(f"  - {dataset}")
    
    print(f"\nModels tested: {len(summary['models_tested'])}")
    for model in summary['models_tested']:
        print(f"  - {model}")
    
    if summary['best_performers']:
        print(f"\n{'='*40}")
        print("BEST PERFORMERS")
        print(f"{'='*40}")
        
        for metric, result in summary['best_performers'].items():
            if result:
                model = result.get('model', 'Unknown')
                dataset = result.get('dataset', 'Unknown')  
                value = result.get(metric.split('_')[-1], 0)
                print(f"{metric:<20}: {model} on {dataset} ({value:.3f})")
    
    if summary['performance_summary']:
        print(f"\n{'='*40}")
        print("PERFORMANCE STATISTICS")
        print(f"{'='*40}")
        
        perf = summary['performance_summary']
        print(f"mAP@0.5     : {perf['map50_stats']['mean']:.3f} ± {perf['map50_stats']['std']:.3f}")
        print(f"FPS         : {perf['fps_stats']['mean']:.1f} ± {perf['fps_stats']['std']:.1f}")
        print(f"Model size  : {perf['model_size_stats']['mean_mb']:.1f} ± {perf['model_size_stats']['std_mb']:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive benchmarking suite')
    parser.add_argument('--datasets', nargs='*', 
                       default=['sample_weeds', 'dummy'],
                       help='Datasets to benchmark')
    parser.add_argument('--models', nargs='*',
                       default=['yolov8n.pt', 'yolov8s.pt', 'yolov11n.pt'],
                       help='Models to benchmark')
    parser.add_argument('--output_dir', default='results/comprehensive_benchmark',
                       help='Output directory')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training benchmark')
    parser.add_argument('--skip_energy', action='store_true', 
                       help='Skip energy benchmark')
    parser.add_argument('--skip_robustness', action='store_true',
                       help='Skip robustness benchmark') 
    parser.add_argument('--existing_models', 
                       help='Directory with existing trained models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Comprehensive Weed Detection Benchmark")
    print(f"📁 Output directory: {output_dir}")
    
    all_results = {}
    
    # Training benchmark
    if not args.skip_training:
        print("\n🏋️ Running training benchmark...")
        training_results = run_training_benchmark(
            datasets=args.datasets,
            models=args.models,
            output_dir=str(output_dir)
        )
        all_results['training'] = training_results
        
        # Save intermediate results
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
    
    # Use existing models if specified
    models_dir = args.existing_models or f"{output_dir}/training"
    
    # Energy benchmark
    if not args.skip_energy and Path(models_dir).exists():
        print("\n⚡ Running energy benchmark...")
        energy_results = run_energy_benchmark(
            models_dir=models_dir,
            data_yaml=f"data/{args.datasets[0]}/data.yaml",
            output_file=str(output_dir / 'energy_results.json')
        )
        all_results['energy'] = energy_results
    
    # Robustness benchmark
    if not args.skip_robustness and Path(models_dir).exists():
        print("\n🛡️ Running robustness benchmark...")
        robustness_results = run_robustness_benchmark(
            models_dir=models_dir,
            data_yaml=f"data/{args.datasets[0]}/data.yaml",
            output_dir=str(output_dir / 'robustness')
        )
        all_results['robustness'] = robustness_results
    
    # Generate comprehensive summary
    if 'training' in all_results:
        summary = generate_summary_report(
            all_results['training'], 
            str(output_dir / 'comprehensive_summary.json')
        )
        
        # Print summary
        print_benchmark_summary(summary)
    
    # Save all results
    with open(output_dir / 'complete_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n🎉 Comprehensive benchmark completed!")
    print(f"📋 Results saved to: {output_dir}")
    print(f"📊 View summary: {output_dir / 'comprehensive_summary.json'}")

if __name__ == '__main__':
    main()
