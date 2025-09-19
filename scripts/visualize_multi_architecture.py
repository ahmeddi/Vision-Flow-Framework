#!/usr/bin/env python3
"""
Multi-Architecture Model Comparison Visualization
=================================================

Generate comprehensive visualization graphs from multi-architecture training results.
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style preferences
plt.style.use('default')
sns.set_palette("husl")

def load_training_summary(summary_file="results/runs/training_summary.json"):
    """Load training summary JSON file"""
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Training summary file {summary_file} not found")
        return []

def create_multi_architecture_comparison():
    """Create comprehensive comparison plots for all architectures"""
    
    # Load data
    data = load_training_summary()
    
    if not data:
        print("No training data available")
        return
    
    # Extract data
    models = []
    architectures = []
    map50 = []
    map50_95 = []
    params = []
    training_times = []
    model_sizes = []
    
    for entry in data:
        model_name = entry['model'].replace('.pt', '')
        models.append(model_name)
        architectures.append(entry['architecture'])
        map50.append(entry['final_map50'])
        map50_95.append(entry['final_map50_95'])
        params.append(entry['total_parameters'] / 1e6)  # Convert to millions
        training_times.append(entry['training_time_seconds'])
        model_sizes.append(entry['model_size_mb'])
    
    # Create color map for architectures
    unique_archs = list(set(architectures))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_archs)))
    arch_colors = {arch: colors[i] for i, arch in enumerate(unique_archs)}
    model_colors = [arch_colors[arch] for arch in architectures]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Multi-Architecture Object Detection Models Comparison\nWeed Detection Performance Analysis', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 1. mAP@0.5 Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(models, map50, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('mAP@0.5 Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mAP@0.5', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{map50[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. mAP@0.5:0.95 Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(models, map50_95, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('mAP@0.5:0.95 Performance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{map50_95[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Model Parameters
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(models, params, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Model Parameters', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Parameters (Millions)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{params[i]:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Training Time
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(models, training_times, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Training Time', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{training_times[i]:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 5. Model Size
    ax5 = plt.subplot(2, 3, 5)
    bars5 = ax5.bar(models, model_sizes, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax5.set_title('Model Size', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Size (MB)', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{model_sizes[i]:.1f}MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Efficiency Plot (mAP vs Parameters)
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(params, map50, c=model_colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    ax6.set_title('Efficiency Analysis', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Parameters (Millions)', fontsize=12)
    ax6.set_ylabel('mAP@0.5', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # Add model labels to scatter points
    for i, model in enumerate(models):
        ax6.annotate(model, (params[i], map50[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=arch_colors[arch], 
                                   alpha=0.8, edgecolor='black', label=arch) 
                      for arch in unique_archs]
    ax6.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
              title='Architecture', title_fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "multi_architecture_comparison.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "multi_architecture_comparison.pdf", 
                bbox_inches='tight', facecolor='white')
    
    print(f"\n📊 Multi-Architecture Comparison saved to:")
    print(f"   PNG: {output_dir}/multi_architecture_comparison.png")
    print(f"   PDF: {output_dir}/multi_architecture_comparison.pdf")
    
    return fig

def create_performance_radar_chart():
    """Create radar chart comparing normalized performance metrics"""
    
    data = load_training_summary()
    if not data:
        return
    
    # Prepare data for radar chart
    models = []
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Speed', 'Efficiency', 'Compactness']
    
    scores = []
    for entry in data:
        model_name = entry['model'].replace('.pt', '')
        models.append(model_name)
        
        # Normalize metrics (0-1 scale)
        map50_norm = entry['final_map50']
        map50_95_norm = entry['final_map50_95']
        speed_norm = min(1.0, 10.0 / entry['training_time_seconds'])  # Faster = higher score
        efficiency_norm = min(1.0, entry['final_map50'] * 10 / (entry['total_parameters'] / 1e6))  # mAP per M params
        compactness_norm = min(1.0, 100.0 / entry['model_size_mb'])  # Smaller = higher score
        
        scores.append([map50_norm, map50_95_norm, speed_norm, efficiency_norm, compactness_norm])
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Set angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, (model, score) in enumerate(zip(models, scores)):
        score += score[:1]  # Complete the circle
        ax.plot(angles, score, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, score, alpha=0.25, color=colors[i])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.title('Multi-Architecture Performance Radar Chart\n(Normalized Metrics)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Save the figure
    output_dir = Path("results/figures")
    plt.savefig(output_dir / "performance_radar_chart.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "performance_radar_chart.pdf", 
                bbox_inches='tight', facecolor='white')
    
    print(f"   Radar Chart: {output_dir}/performance_radar_chart.png")
    
    return fig

def create_summary_table():
    """Create a detailed summary table"""
    
    data = load_training_summary()
    if not data:
        return
    
    # Create DataFrame
    df_data = []
    for entry in data:
        df_data.append({
            'Model': entry['model'].replace('.pt', ''),
            'Architecture': entry['architecture'],
            'mAP@0.5': f"{entry['final_map50']:.3f}",
            'mAP@0.5:0.95': f"{entry['final_map50_95']:.3f}",
            'Parameters (M)': f"{entry['total_parameters']/1e6:.1f}",
            'Size (MB)': f"{entry['model_size_mb']:.1f}",
            'Training Time (s)': f"{entry['training_time_seconds']:.1f}",
            'Efficiency*': f"{entry['final_map50'] * 10 / (entry['total_parameters']/1e6):.2f}"
        })
    
    df = pd.DataFrame(df_data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Multi-Architecture Model Comparison Summary\n*Efficiency = mAP@0.5 × 10 / Parameters(M)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save the figure
    output_dir = Path("results/figures")
    plt.savefig(output_dir / "summary_table.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "summary_table.pdf", 
                bbox_inches='tight', facecolor='white')
    
    print(f"   Summary Table: {output_dir}/summary_table.png")
    
    return fig

def main():
    """Generate all visualization graphs"""
    
    print("🎨 Generating Multi-Architecture Visualization Graphs...")
    print("=" * 60)
    
    # Create figures directory
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    fig1 = create_multi_architecture_comparison()
    fig2 = create_performance_radar_chart() 
    fig3 = create_summary_table()
    
    print("=" * 60)
    print("✅ All visualization graphs generated successfully!")
    print("\n📁 Generated Files:")
    print("   📊 Multi-Architecture Comparison (PNG & PDF)")
    print("   🎯 Performance Radar Chart (PNG & PDF)")
    print("   📋 Summary Table (PNG & PDF)")
    
    return [fig1, fig2, fig3]

if __name__ == "__main__":
    main()
    # Show plots (optional)
    # plt.show()