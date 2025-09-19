#!/usr/bin/env python3
"""
Fixed Comprehensive Multi-Model Multi-Dataset Training Script
============================================================
Train all models on all datasets systematically.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_training_command(models, dataset, epochs=1):
    """Run training command for given models and dataset."""
    
    # Build command as a string for shell execution
    models_str = ' '.join(models)
    cmd = f'python scripts/train.py --models {models_str} --data data/{dataset}.yaml --epochs {epochs} --config configs/base.yaml'
    
    print(f'🏃 Running: {cmd}')
    
    try:
        # Use shell=True for Windows compatibility
        result = subprocess.run(cmd, shell=True, check=True, timeout=7200)
        return True, "SUCCESS"
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except subprocess.CalledProcessError as e:
        return False, f"FAILED (exit code {e.returncode})"
    except Exception as e:
        return False, f"ERROR ({str(e)})"

def main():
    """Main training function."""
    
    # Define all models and datasets
    models = ['yolov8n.pt', 'yolo11n.pt', 'yolox_s', 'efficientdet_d0', 'detr_resnet50']
    datasets = ['deepweeds', 'weed25', 'weedsgalore', 'cwd30']
    
    print('🚀 Starting comprehensive multi-model, multi-dataset training...')
    print(f'📊 Training {len(models)} models on {len(datasets)} datasets')
    print(f'🔥 Total combinations: {len(models) * len(datasets)} training runs')
    print('⏰ Testing with 1 epoch each for quick validation!')
    print('=' * 60)
    
    results = []
    total_start_time = time.time()
    
    for i, dataset in enumerate(datasets, 1):
        print(f'\\n📂 [{i}/{len(datasets)}] Starting training on {dataset} dataset...')
        
        start_time = time.time()
        success, status = run_training_command(models, dataset, epochs=1)
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        if success:
            print(f'✅ {dataset} training completed in {elapsed_min:.1f} minutes')
            results.append(f'✅ {dataset}: {status} ({elapsed_min:.1f}min)')
        else:
            print(f'❌ {dataset} training {status}')
            results.append(f'❌ {dataset}: {status}')
    
    total_elapsed = time.time() - total_start_time
    total_hours = total_elapsed / 3600
    
    print('\\n' + '=' * 60)
    print('🎉 ALL TRAINING COMPLETED!')
    print(f'⏰ Total time: {total_hours:.1f} hours')
    print('📊 SUMMARY:')
    for result in results:
        print(f'   {result}')
    print('=' * 60)
    
    # Count successes
    successes = sum(1 for r in results if '✅' in r)
    print(f'🎯 Success rate: {successes}/{len(datasets)} datasets ({successes/len(datasets)*100:.1f}%)')
    
    # Run statistical analysis if all training completed successfully
    if successes == len(datasets):
        print('\\n' + '=' * 60)
        print('📊 Starting statistical analysis...')
        print('=' * 60)
        
        try:
            # Run statistical analysis script
            analysis_cmd = 'python scripts/statistical_analysis.py'
            print(f'🔬 Running: {analysis_cmd}')
            
            result = subprocess.run(analysis_cmd, shell=True, check=True, timeout=600)
            print('✅ Statistical analysis completed successfully!')
            
            # Run visualization generation
            print('\\n📈 Generating comparison visualizations...')
            viz_cmd = 'python scripts/create_visualizations.py'
            print(f'🎨 Running: {viz_cmd}')
            
            result = subprocess.run(viz_cmd, shell=True, check=True, timeout=600)
            print('✅ Visualization generation completed successfully!')
            
        except subprocess.TimeoutExpired:
            print('⏰ Analysis/visualization timed out (10 minutes each)')
        except subprocess.CalledProcessError as e:
            print(f'❌ Analysis/visualization failed (exit code {e.returncode})')
        except Exception as e:
            print(f'💥 Analysis/visualization error: {str(e)}')
    else:
        print('\\n⚠️  Statistical analysis skipped due to training failures')
        print('   All datasets must complete successfully to run analysis')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\\n⚠️ Training interrupted by user')
    except Exception as e:
        print(f'\\n💥 Training failed with error: {e}')
        sys.exit(1)