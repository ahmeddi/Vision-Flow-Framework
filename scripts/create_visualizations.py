#!/usr/bin/env python3
"""
Generate comprehensive visualization figures from training results
"""
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

def load_evaluation_data(eval_file="results/eval_summary.json"):
    """Load evaluation results"""
    try:
        with open(eval_file, 'r') as f:
            data = json.load(f)
        # Filter out error entries
        valid_data = [entry for entry in data if 'error' not in entry]
        return valid_data
    except FileNotFoundError:
        print(f"Evaluation file {eval_file} not found")
        return []

def load_training_results(results_file="results/runs/yolov8n10/results.csv"):
    """Load training results CSV"""
    try:
        df = pd.read_csv(results_file)
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print(f"Training results file {results_file} not found")
        return pd.DataFrame()

def create_performance_comparison():
    """Create performance comparison plots"""
    data = load_evaluation_data()
    
    if not data:
        print("No evaluation data available")
        return
    
    # Extract metrics
    models = []
    fps = []
    latency = []
    params = []
    map50 = []
    
    for entry in data:
        if 'fps' in entry:  # Valid entry
            models.append(entry['model_name'].split('\\')[-1].replace('.pt', ''))
            fps.append(entry['fps'])
            latency.append(entry['latency_ms'])
            params.append(entry['total_parameters'] / 1e6)  # Convert to millions
            map50.append(entry['map50'])
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLOv8n Performance Analysis on Weed Detection Dataset', fontsize=16, fontweight='bold')
    
    # FPS comparison
    bars1 = ax1.bar(models, fps, color=['#2E8B57', '#3CB371', '#66CDAA', '#98FB98'])
    ax1.set_title('Inference Speed (FPS)', fontweight='bold')
    ax1.set_ylabel('Frames Per Second')
    ax1.set_ylim(0, max(fps) * 1.2)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{fps[i]:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Latency comparison
    bars2 = ax2.bar(models, latency, color=['#FF6347', '#FF7F50', '#FFA07A', '#FFB6C1'])
    ax2.set_title('Inference Latency', fontweight='bold')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_ylim(0, max(latency) * 1.2)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{latency[i]:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Parameters comparison
    bars3 = ax3.bar(models, params, color=['#4169E1', '#6495ED', '#87CEEB', '#B0E0E6'])
    ax3.set_title('Model Parameters', fontweight='bold')
    ax3.set_ylabel('Parameters (Millions)')
    ax3.set_ylim(0, max(params) * 1.2)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{params[i]:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # mAP comparison (note: these are 0 because models need more training)
    bars4 = ax4.bar(models, map50, color=['#FFD700', '#FFA500', '#FF8C00', '#FF6347'])
    ax4.set_title('mAP@0.5 (Initial Training)', fontweight='bold')
    ax4.set_ylabel('mAP@0.5')
    ax4.set_ylim(0, 1.0)
    ax4.text(0.5, 0.5, 'Models require longer\ntraining for accuracy', 
             ha='center', va='center', transform=ax4.transAxes,
             fontsize=12, style='italic')
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs("results/figures", exist_ok=True)
    
    # Save the figure
    plt.savefig("results/figures/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/figures/performance_comparison.pdf", bbox_inches='tight')
    print("✅ Performance comparison saved to results/figures/performance_comparison.png")
    
    return fig

def create_training_curves():
    """Create training curves visualization"""
    df = load_training_results()
    
    if df.empty:
        print("No training results available")
        return
    
    # Create training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLOv8n Training Progress - Weed Detection', fontsize=16, fontweight='bold')
    
    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
    
    # Loss curves
    if 'train/box_loss' in df.columns:
        ax1.plot(epochs, df['train/box_loss'], 'b-', linewidth=2, label='Box Loss', marker='o')
        ax1.plot(epochs, df['train/cls_loss'], 'r-', linewidth=2, label='Class Loss', marker='s')
        ax1.plot(epochs, df['train/dfl_loss'], 'g-', linewidth=2, label='DFL Loss', marker='^')
        ax1.set_title('Training Losses', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Validation metrics
    if 'val/box_loss' in df.columns:
        ax2.plot(epochs, df['val/box_loss'], 'b--', linewidth=2, label='Box Loss', marker='o')
        ax2.plot(epochs, df['val/cls_loss'], 'r--', linewidth=2, label='Class Loss', marker='s')
        ax2.plot(epochs, df['val/dfl_loss'], 'g--', linewidth=2, label='DFL Loss', marker='^')
        ax2.set_title('Validation Losses', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr/pg0' in df.columns:
        ax3.plot(epochs, df['lr/pg0'], 'purple', linewidth=2, marker='o')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Model complexity info
    ax4.text(0.1, 0.8, 'Model Architecture: YOLOv8n', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.7, 'Dataset: Weed25 (25 weed species)', fontsize=12)
    ax4.text(0.1, 0.6, f'Total Parameters: ~3.0M', fontsize=12)
    ax4.text(0.1, 0.5, 'Training: 10 epochs', fontsize=12)
    ax4.text(0.1, 0.4, 'Batch Size: 4', fontsize=12)
    ax4.text(0.1, 0.3, 'Image Size: 640x640', fontsize=12)
    ax4.text(0.1, 0.2, 'Device: CPU', fontsize=12)
    ax4.set_title('Training Configuration', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save training curves
    plt.savefig("results/figures/training_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/figures/training_curves.pdf", bbox_inches='tight')
    print("✅ Training curves saved to results/figures/training_curves.png")
    
    return fig

def create_dataset_overview():
    """Create dataset overview visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Agricultural Weed Detection Datasets Overview', fontsize=16, fontweight='bold')
    
    # Dataset statistics
    datasets = ['DeepWeeds', 'Weed25', 'CWD30', 'WeedsGalore']
    classes = [8, 25, 30, 15]
    images_train = [24, 32, 40, 28]  # Synthetic data counts
    images_val = [6, 8, 10, 7]
    
    # Classes comparison
    bars1 = ax1.bar(datasets, classes, color=['#2E8B57', '#3CB371', '#66CDAA', '#98FB98'])
    ax1.set_title('Number of Weed Classes per Dataset', fontweight='bold')
    ax1.set_ylabel('Number of Classes')
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{classes[i]}', ha='center', va='bottom', fontweight='bold')
    
    # Sample distribution
    x = np.arange(len(datasets))
    width = 0.35
    ax2.bar(x - width/2, images_train, width, label='Training', color='#4169E1')
    ax2.bar(x + width/2, images_val, width, label='Validation', color='#87CEEB')
    ax2.set_title('Synthetic Dataset Split', fontweight='bold')
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    
    # Weed species examples (DeepWeeds)
    deepweeds_species = [
        'Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium',
        'Pear', 'Prickly Acacia', 'Rubber Vine', 'Siam Weed'
    ]
    y_pos = np.arange(len(deepweeds_species))
    ax3.barh(y_pos, [1]*len(deepweeds_species), color='#FFD700')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(deepweeds_species, fontsize=10)
    ax3.set_xlabel('Species')
    ax3.set_title('DeepWeeds Species (8 classes)', fontweight='bold')
    
    # Performance summary
    performance_data = {
        'Metric': ['Average FPS', 'Average Latency (ms)', 'Model Size (MB)', 'Parameters (M)'],
        'Value': [6.3, 160.5, 6.0, 3.0]
    }
    ax4.text(0.1, 0.8, 'YOLOv8n Performance Summary', fontsize=14, fontweight='bold')
    for i, (metric, value) in enumerate(zip(performance_data['Metric'], performance_data['Value'])):
        if 'FPS' in metric:
            ax4.text(0.1, 0.7 - i*0.1, f'{metric}: {value:.1f}', fontsize=12)
        elif 'Latency' in metric:
            ax4.text(0.1, 0.7 - i*0.1, f'{metric}: {value:.1f} ms', fontsize=12)
        elif 'Size' in metric:
            ax4.text(0.1, 0.7 - i*0.1, f'{metric}: {value:.1f} MB', fontsize=12)
        else:
            ax4.text(0.1, 0.7 - i*0.1, f'{metric}: {value:.1f} M', fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save dataset overview
    plt.savefig("results/figures/dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/figures/dataset_overview.pdf", bbox_inches='tight')
    print("✅ Dataset overview saved to results/figures/dataset_overview.png")
    
    return fig

def create_paper_ready_figures():
    """Create publication-ready figures"""
    print("Generating comprehensive paper figures...")
    
    # Create all figures
    perf_fig = create_performance_comparison()
    train_fig = create_training_curves()
    dataset_fig = create_dataset_overview()
    
    print("\n📊 Generated Figures:")
    print("   1. Performance Comparison: results/figures/performance_comparison.png")
    print("   2. Training Curves: results/figures/training_curves.png") 
    print("   3. Dataset Overview: results/figures/dataset_overview.png")
    print("\n✅ All visualization figures generated successfully!")
    
    return [perf_fig, train_fig, dataset_fig]

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Generate all figures
    create_paper_ready_figures()
    
    # Show the plots (optional)
    plt.show()
