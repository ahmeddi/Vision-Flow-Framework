#!/usr/bin/env python3
"""
Comprehensive Multi-Model Multi-Dataset Training Script
======================================================
Train all models on all datasets systematically.
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    # Define all models and datasets
    models = ['yolov8n.pt', 'yolo11n.pt', 'yolox_s', 'efficientdet_d0', 'detr_resnet50']
    datasets = ['deepweeds', 'weed25', 'weedsgalore', 'cwd30']
    
    print('🚀 Starting comprehensive multi-model, multi-dataset training...')
    print(f'📊 Training {len(models)} models on {len(datasets)} datasets')
    print(f'🔥 Total combinations: {len(models) * len(datasets)} training runs')
    print('⏰ This will take several hours with 50 epochs each!')
    print('='*60)
    
    results = []
    
    for i, dataset in enumerate(datasets, 1):
        print(f'\n📂 [{i}/{len(datasets)}] Starting training on {dataset} dataset...')
        
        cmd = [
            'python', 'scripts/train.py',
            '--models'] + models + [
            '--data', f'data/{dataset}.yaml',
            '--epochs', '50',
            '--config', 'configs/base.yaml'
        ]
        
        cmd_str = ' '.join(cmd)
        print(f'🏃 Command: {cmd_str}')
        start_time = time.time()
        
        try:
            # Run training for this dataset
            result = subprocess.run(cmd, check=True, timeout=7200, shell=True)  # 2 hour timeout
            
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60
            print(f'✅ {dataset} training completed in {elapsed_min:.1f} minutes')
            results.append(f'✅ {dataset}: SUCCESS ({elapsed_min:.1f}min)')
            
        except subprocess.TimeoutExpired:
            print(f'⏰ {dataset} training timed out after 2 hours')
            results.append(f'⏰ {dataset}: TIMEOUT')
            
        except subprocess.CalledProcessError as e:
            print(f'❌ {dataset} training failed with exit code {e.returncode}')
            results.append(f'❌ {dataset}: FAILED')
            
        except Exception as e:
            print(f'💥 {dataset} training error: {e}')
            results.append(f'💥 {dataset}: ERROR')
    
    print('\n' + '='*60)
    print('🎉 ALL TRAINING COMPLETED!')
    print('📊 SUMMARY:')
    for result in results:
        print(f'   {result}')
    print('='*60)
    
    # Count successes
    successes = sum(1 for r in results if '✅' in r)
    print(f'🎯 Success rate: {successes}/{len(datasets)} datasets ({successes/len(datasets)*100:.1f}%)')
    
    # Run statistical analysis if all training completed successfully
    if successes == len(datasets):
        print('\n' + '='*60)
        print('📊 Starting statistical analysis...')
        print('='*60)
        
        try:
            # Run statistical analysis script
            analysis_cmd = 'python scripts/statistical_analysis.py'
            print(f'🔬 Running: {analysis_cmd}')
            
            result = subprocess.run(analysis_cmd, shell=True, check=True, timeout=600)
            print('✅ Statistical analysis completed successfully!')
            
            # Run visualization generation
            print('\n📈 Generating comparison visualizations...')
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
        print('\n⚠️  Statistical analysis skipped due to training failures')
        print('   All datasets must complete successfully to run analysis')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n⚠️ Training interrupted by user')
    except Exception as e:
        print(f'\n💥 Training failed with error: {e}')
        sys.exit(1)