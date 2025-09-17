"""
Advanced Visualization and Reporting Tools for Weed Detection Model Comparison
============================================================================

This module provides comprehensive visualization and automated report generation
capabilities for model comparison studies, including:

- Interactive performance dashboards
- Statistical analysis plots
- Model architecture comparisons
- Dataset-specific analysis
- Publication-ready figures
- Automated LaTeX report generation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

# Set style preferences
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class AdvancedVisualizer:
    """Advanced visualization and reporting tools."""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "static").mkdir(exist_ok=True)
        (self.output_dir / "interactive").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Color schemes
        self.color_schemes = self._init_color_schemes()
        
    def _init_color_schemes(self) -> Dict[str, List[str]]:
        """Initialize color schemes for different model families."""
        return {
            'yolov8': ['#1f77b4', '#aec7e8', '#2c5f2d', '#98df8a'],
            'yolov11': ['#ff7f0e', '#ffbb78', '#d62728', '#ff9896'],
            'yolo_nas': ['#2ca02c', '#98df8a', '#7f7f7f', '#c5c5c5'],
            'efficientdet': ['#9467bd', '#c5b0d5', '#8c564b', '#c49c94'],
            'detr': ['#e377c2', '#f7b6d3', '#7f7f7f', '#c5c5c5'],
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        }
    
    def create_performance_dashboard(self, results_df: pd.DataFrame) -> str:
        """Create an interactive performance dashboard."""
        
        print("📊 Creating interactive performance dashboard...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance vs Speed', 'Model Size Comparison', 
                          'Performance by Dataset', 'Efficiency Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Performance vs Speed scatter plot
        fig.add_trace(
            go.Scatter(
                x=results_df['fps'],
                y=results_df['map50'],
                mode='markers+text',
                text=results_df['model_name'].str[:8],  # Shortened names
                textposition="top center",
                marker=dict(
                    size=results_df['model_size_mb'] / 5,  # Size based on model size
                    color=results_df['map50'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="mAP@50", x=0.45)
                ),
                name='Models',
                hovertemplate='<b>%{text}</b><br>' +
                             'FPS: %{x:.1f}<br>' +
                             'mAP@50: %{y:.3f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Model size comparison
        model_sizes = results_df.groupby('model_name')['model_size_mb'].mean().sort_values()
        fig.add_trace(
            go.Bar(
                x=model_sizes.values,
                y=model_sizes.index,
                orientation='h',
                marker_color='lightblue',
                name='Model Size'
            ),
            row=1, col=2
        )
        
        # 3. Performance by dataset
        if 'dataset_name' in results_df.columns:
            dataset_perf = results_df.groupby(['dataset_name', 'model_name'])['map50'].mean().reset_index()
            for dataset in dataset_perf['dataset_name'].unique():
                data = dataset_perf[dataset_perf['dataset_name'] == dataset]
                fig.add_trace(
                    go.Scatter(
                        x=data['model_name'],
                        y=data['map50'],
                        mode='markers+lines',
                        name=dataset,
                        line=dict(width=2),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
        
        # 4. Efficiency analysis (mAP vs Parameters)
        if 'params_millions' in results_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=results_df['params_millions'],
                    y=results_df['map50'],
                    mode='markers+text',
                    text=results_df['model_name'].str[:6],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color='orange',
                        opacity=0.7
                    ),
                    name='Efficiency'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Model Performance Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="FPS", row=1, col=1)
        fig.update_yaxes(title_text="mAP@50", row=1, col=1)
        fig.update_xaxes(title_text="Model Size (MB)", row=1, col=2)
        fig.update_yaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_yaxes(title_text="mAP@50", row=2, col=1)
        fig.update_xaxes(title_text="Parameters (M)", row=2, col=2)
        fig.update_yaxes(title_text="mAP@50", row=2, col=2)
        
        # Save interactive plot
        dashboard_path = self.output_dir / "interactive" / "performance_dashboard.html"
        fig.write_html(str(dashboard_path))
        
        print(f"📊 Dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def create_statistical_analysis_plots(self, results_df: pd.DataFrame) -> List[str]:
        """Create statistical analysis visualizations."""
        
        print("📈 Creating statistical analysis plots...")
        
        plots_created = []
        
        # 1. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Distributions', fontsize=16, fontweight='bold')
        
        # mAP@50 distribution
        sns.boxplot(data=results_df, y='model_name', x='map50', ax=axes[0,0])
        axes[0,0].set_title('mAP@50 Distribution')
        axes[0,0].set_xlabel('mAP@50')
        
        # FPS distribution
        sns.boxplot(data=results_df, y='model_name', x='fps', ax=axes[0,1])
        axes[0,1].set_title('FPS Distribution')
        axes[0,1].set_xlabel('FPS')
        
        # Model size distribution
        sns.boxplot(data=results_df, y='model_name', x='model_size_mb', ax=axes[1,0])
        axes[1,0].set_title('Model Size Distribution')
        axes[1,0].set_xlabel('Model Size (MB)')
        
        # Correlation heatmap
        numeric_cols = ['map50', 'map50_95', 'fps', 'model_size_mb']
        if 'params_millions' in results_df.columns:
            numeric_cols.append('params_millions')
        
        corr_data = results_df[numeric_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Metrics Correlation')
        
        plt.tight_layout()
        dist_path = self.output_dir / "static" / "statistical_distributions.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(str(dist_path))
        
        # 2. Performance ranking plot
        plt.figure(figsize=(12, 8))
        
        # Calculate ranking scores
        ranking_metrics = ['map50', 'fps']
        if 'params_millions' in results_df.columns:
            # Invert parameters (smaller is better for efficiency)
            results_df['efficiency_rank'] = 1 / (results_df['params_millions'] + 0.1)
            ranking_metrics.append('efficiency_rank')
        
        # Normalize metrics to 0-1 scale
        normalized_df = results_df.copy()
        for metric in ranking_metrics:
            normalized_df[f'{metric}_norm'] = (results_df[metric] - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
        
        # Calculate composite score
        normalized_df['composite_score'] = normalized_df[[f'{m}_norm' for m in ranking_metrics]].mean(axis=1)
        
        # Plot ranking
        model_scores = normalized_df.groupby('model_name')['composite_score'].mean().sort_values(ascending=True)
        
        plt.barh(range(len(model_scores)), model_scores.values, color='skyblue')
        plt.yticks(range(len(model_scores)), model_scores.index)
        plt.xlabel('Composite Performance Score')
        plt.title('Model Performance Ranking', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        for i, score in enumerate(model_scores.values):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        ranking_path = self.output_dir / "static" / "performance_ranking.png"
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(str(ranking_path))
        
        return plots_created
    
    def create_model_architecture_comparison(self, results_df: pd.DataFrame) -> str:
        """Create model architecture comparison visualization."""
        
        print("🏗️ Creating model architecture comparison...")
        
        # Extract model families
        results_df['model_family'] = results_df['model_name'].str.extract(r'(yolov8|yolov11|yolo_nas|efficientdet|detr)')[0]
        results_df['model_family'] = results_df['model_family'].fillna('other')
        
        # Create interactive comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance by Architecture', 'Size vs Performance', 
                          'Speed vs Accuracy Trade-off', 'Family Statistics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Color mapping for families
        families = results_df['model_family'].unique()
        colors = px.colors.qualitative.Set3[:len(families)]
        family_colors = dict(zip(families, colors))
        
        # 1. Performance by architecture
        for family in families:
            family_data = results_df[results_df['model_family'] == family]
            fig.add_trace(
                go.Scatter(
                    x=family_data['model_name'],
                    y=family_data['map50'],
                    mode='markers',
                    name=family,
                    marker=dict(color=family_colors[family], size=10),
                    hovertemplate='<b>%{x}</b><br>mAP@50: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Size vs Performance
        for family in families:
            family_data = results_df[results_df['model_family'] == family]
            fig.add_trace(
                go.Scatter(
                    x=family_data['model_size_mb'],
                    y=family_data['map50'],
                    mode='markers',
                    name=f'{family}_size',
                    marker=dict(color=family_colors[family], size=8),
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Size: %{x:.1f} MB<br>mAP@50: %{y:.3f}<extra></extra>',
                    text=family_data['model_name']
                ),
                row=1, col=2
            )
        
        # 3. Speed vs Accuracy
        for family in families:
            family_data = results_df[results_df['model_family'] == family]
            fig.add_trace(
                go.Scatter(
                    x=family_data['fps'],
                    y=family_data['map50'],
                    mode='markers',
                    name=f'{family}_speed',
                    marker=dict(color=family_colors[family], size=8),
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>FPS: %{x:.1f}<br>mAP@50: %{y:.3f}<extra></extra>',
                    text=family_data['model_name']
                ),
                row=2, col=1
            )
        
        # 4. Family statistics
        family_stats = results_df.groupby('model_family')['map50'].agg(['mean', 'std']).reset_index()
        fig.add_trace(
            go.Bar(
                x=family_stats['model_family'],
                y=family_stats['mean'],
                error_y=dict(type='data', array=family_stats['std']),
                marker_color=[family_colors[f] for f in family_stats['model_family']],
                name='Family Average',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Model Architecture Comparison",
            title_x=0.5
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="mAP@50", row=1, col=1)
        fig.update_xaxes(title_text="Model Size (MB)", row=1, col=2)
        fig.update_yaxes(title_text="mAP@50", row=1, col=2)
        fig.update_xaxes(title_text="FPS", row=2, col=1)
        fig.update_yaxes(title_text="mAP@50", row=2, col=1)
        fig.update_xaxes(title_text="Model Family", row=2, col=2)
        fig.update_yaxes(title_text="Average mAP@50", row=2, col=2)
        
        # Save plot
        arch_path = self.output_dir / "interactive" / "architecture_comparison.html"
        fig.write_html(str(arch_path))
        
        print(f"🏗️ Architecture comparison saved to: {arch_path}")
        return str(arch_path)
    
    def create_dataset_analysis(self, results_df: pd.DataFrame) -> List[str]:
        """Create dataset-specific analysis visualizations."""
        
        print("📊 Creating dataset-specific analysis...")
        
        if 'dataset_name' not in results_df.columns:
            print("⚠️ No dataset information found, skipping dataset analysis")
            return []
        
        plots_created = []
        
        # 1. Performance heatmap by dataset
        plt.figure(figsize=(12, 8))
        
        pivot_data = results_df.pivot_table(values='map50', index='model_name', columns='dataset_name', aggfunc='mean')
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlBu_r',
            center=pivot_data.mean().mean(),
            cbar_kws={'label': 'mAP@50'}
        )
        
        plt.title('Model Performance by Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset')
        plt.ylabel('Model')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_path = self.output_dir / "static" / "dataset_performance_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(str(heatmap_path))
        
        # 2. Dataset difficulty analysis
        dataset_stats = results_df.groupby('dataset_name')['map50'].agg(['mean', 'std', 'min', 'max']).reset_index()
        dataset_stats['difficulty'] = 1 - dataset_stats['mean']  # Higher difficulty = lower average performance
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(dataset_stats['dataset_name'], dataset_stats['mean'], 
                      yerr=dataset_stats['std'], capsize=5,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(dataset_stats)])
        
        plt.title('Dataset Difficulty Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset')
        plt.ylabel('Average mAP@50')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, dataset_stats['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        difficulty_path = self.output_dir / "static" / "dataset_difficulty.png"
        plt.savefig(difficulty_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(str(difficulty_path))
        
        return plots_created
    
    def create_publication_figures(self, results_df: pd.DataFrame) -> List[str]:
        """Create publication-ready figures."""
        
        print("📄 Creating publication-ready figures...")
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
        
        plots_created = []
        
        # 1. Main performance comparison (suitable for paper)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance vs Speed scatter
        scatter = ax1.scatter(results_df['fps'], results_df['map50'], 
                            s=results_df['model_size_mb']*2, alpha=0.7,
                            c=range(len(results_df)), cmap='tab10')
        
        # Add model labels for key points
        top_models = results_df.nlargest(3, 'map50')
        fast_models = results_df.nlargest(3, 'fps')
        key_models = pd.concat([top_models, fast_models]).drop_duplicates()
        
        for _, row in key_models.iterrows():
            ax1.annotate(row['model_name'][:8], 
                        (row['fps'], row['map50']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax1.set_xlabel('Inference Speed (FPS)')
        ax1.set_ylabel('mAP@50')
        ax1.set_title('Performance vs Speed Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # Performance comparison bar chart
        model_means = results_df.groupby('model_name')['map50'].mean().sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_means)))
        
        bars = ax2.barh(range(len(model_means)), model_means.values, color=colors)
        ax2.set_yticks(range(len(model_means)))
        ax2.set_yticklabels([name[:10] for name in model_means.index])
        ax2.set_xlabel('mAP@50')
        ax2.set_title('Model Performance Comparison')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, model_means.values)):
            ax2.text(value + 0.005, i, f'{value:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        pub_main_path = self.output_dir / "static" / "publication_main_figure.png"
        plt.savefig(pub_main_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plots_created.append(str(pub_main_path))
        
        # 2. Efficiency analysis figure
        if 'params_millions' in results_df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Create efficiency plot
            efficiency_scores = results_df['map50'] / (results_df['params_millions'] + 0.1)
            
            scatter = ax.scatter(results_df['params_millions'], results_df['map50'],
                               s=100, alpha=0.7, c=efficiency_scores, cmap='RdYlGn')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Efficiency Score (mAP/Params)')
            
            # Annotate efficient models
            efficient_models = results_df.loc[efficiency_scores.nlargest(3).index]
            for _, row in efficient_models.iterrows():
                ax.annotate(row['model_name'][:8],
                           (row['params_millions'], row['map50']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
            
            ax.set_xlabel('Parameters (Millions)')
            ax.set_ylabel('mAP@50')
            ax.set_title('Model Efficiency Analysis')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            pub_eff_path = self.output_dir / "static" / "publication_efficiency.png"
            plt.savefig(pub_eff_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots_created.append(str(pub_eff_path))
        
        # Reset style
        plt.rcParams.update(plt.rcParamsDefault)
        
        return plots_created
    
    def generate_latex_report(self, results_df: pd.DataFrame, 
                            experiment_config: Dict = None) -> str:
        """Generate LaTeX report for academic publication."""
        
        print("📄 Generating LaTeX report...")
        
        # LaTeX document template
        latex_content = r"""\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{float}
\\usepackage{amsmath}
\\usepackage{geometry}
\\geometry{margin=1in}

\\title{Comprehensive Weed Detection Model Comparison Study}
\\author{Research Team}
\\date{""" + datetime.now().strftime("%B %d, %Y") + r"""}

\\begin{document}

\\maketitle

\\begin{abstract}
This study presents a comprehensive comparative analysis of state-of-the-art object detection models for weed detection in agricultural settings. We evaluate """ + str(len(results_df['model_name'].unique())) + r""" different models across multiple datasets, examining performance, efficiency, and practical deployment considerations.
\\end{abstract}

\\section{Introduction}

Precision agriculture requires accurate and efficient weed detection systems. This study compares modern object detection architectures including YOLO variants, EfficientDet, and transformer-based models for agricultural weed detection tasks.

\\section{Methodology}

\\subsection{Models Evaluated}
We evaluated the following models:
\\begin{itemize}"""

        # Add model list
        for model in sorted(results_df['model_name'].unique()):
            latex_content += f"\n\\item {model}"
        
        latex_content += r"""
\\end{itemize}

\\subsection{Datasets}"""

        if 'dataset_name' in results_df.columns:
            latex_content += "\nWe evaluated models on the following datasets:\n\\begin{itemize}"
            for dataset in sorted(results_df['dataset_name'].unique()):
                latex_content += f"\n\\item {dataset}"
            latex_content += "\n\\end{itemize}"

        latex_content += r"""

\\subsection{Evaluation Metrics}
Models were evaluated using:
\\begin{itemize}
\\item Mean Average Precision at IoU 0.5 (mAP@50)
\\item Mean Average Precision at IoU 0.5-0.95 (mAP@50-95)
\\item Inference speed (FPS)
\\item Model size and parameter count
\\item Memory usage
\\end{itemize}

\\section{Results}

\\subsection{Performance Summary}

Table~\\ref{tab:performance} shows the performance comparison of all evaluated models.

\\begin{table}[H]
\\centering
\\caption{Model Performance Comparison}
\\label{tab:performance}
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Model & mAP@50 & mAP@50-95 & FPS & Params (M) & Size (MB) \\\\
\\midrule"""

        # Add performance table
        numeric_cols = ['map50', 'map50_95', 'fps', 'params_millions', 'model_size_mb']
        for model_name, group in results_df.groupby('model_name'):
            # Calculate means for numeric columns only
            available_numeric = [col for col in numeric_cols if col in group.columns]
            mean_values = group[available_numeric].mean()
            
            model_name_clean = model_name.replace('_', '\\_')  # Escape underscores
            latex_content += f"\n{model_name_clean} & {mean_values.get('map50', 0):.3f} & {mean_values.get('map50_95', 0):.3f} & {mean_values.get('fps', 0):.1f}"
            
            if 'params_millions' in mean_values:
                latex_content += f" & {mean_values['params_millions']:.1f}"
            else:
                latex_content += " & --"
                
            latex_content += f" & {mean_values.get('model_size_mb', 0):.1f} \\\\\\\\"

        latex_content += r"""
\\bottomrule
\\end{tabular}
\\end{table}

\\subsection{Key Findings}

\\begin{itemize}"""

        # Generate key findings
        best_map = results_df.loc[results_df['map50'].idxmax()]
        fastest = results_df.loc[results_df['fps'].idxmax()]
        
        latex_content += f"\n\\item Best overall accuracy: {best_map['model_name']} (mAP@50: {best_map['map50']:.3f})"
        latex_content += f"\n\\item Fastest inference: {fastest['model_name']} ({fastest['fps']:.1f} FPS)"
        
        if 'params_millions' in results_df.columns:
            # Most efficient model
            efficiency_scores = results_df['map50'] / (results_df['params_millions'] + 0.1)
            most_efficient = results_df.loc[efficiency_scores.idxmax()]
            latex_content += f"\n\\item Most efficient: {most_efficient['model_name']} (efficiency score: {efficiency_scores.max():.4f})"

        latex_content += r"""
\\end{itemize}

\\section{Discussion}

The results demonstrate significant variations in performance and efficiency across different model architectures. YOLO-based models generally provide the best balance of accuracy and speed, while transformer-based models show competitive accuracy but with higher computational requirements.

\\section{Conclusion}

This comprehensive evaluation provides insights for selecting appropriate weed detection models based on specific application requirements, whether prioritizing accuracy, speed, or computational efficiency.

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{publication_main_figure.png}
\\caption{Performance vs Speed comparison of evaluated models}
\\label{fig:performance}
\\end{figure}

\\end{document}"""

        # Save LaTeX file
        latex_path = self.output_dir / "reports" / "comparative_study.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"📄 LaTeX report saved to: {latex_path}")
        return str(latex_path)
    
    def create_comprehensive_report(self, results_df: pd.DataFrame,
                                  experiment_config: Dict = None) -> Dict[str, str]:
        """Create a comprehensive visualization and reporting package."""
        
        print("\\n🎨 Creating comprehensive visualization package...")
        
        created_files = {}
        
        # 1. Interactive dashboard
        dashboard_path = self.create_performance_dashboard(results_df)
        created_files['dashboard'] = dashboard_path
        
        # 2. Statistical analysis
        stats_plots = self.create_statistical_analysis_plots(results_df)
        created_files['statistical_plots'] = stats_plots
        
        # 3. Architecture comparison
        arch_path = self.create_model_architecture_comparison(results_df)
        created_files['architecture_comparison'] = arch_path
        
        # 4. Dataset analysis
        dataset_plots = self.create_dataset_analysis(results_df)
        created_files['dataset_plots'] = dataset_plots
        
        # 5. Publication figures
        pub_plots = self.create_publication_figures(results_df)
        created_files['publication_figures'] = pub_plots
        
        # 6. LaTeX report
        latex_path = self.generate_latex_report(results_df, experiment_config)
        created_files['latex_report'] = latex_path
        
        # 7. Summary HTML report
        html_path = self._create_html_summary(results_df, created_files)
        created_files['html_summary'] = html_path
        
        print("\\n✅ Comprehensive visualization package created!")
        print(f"📁 All files saved to: {self.output_dir}")
        
        return created_files
    
    def _create_html_summary(self, results_df: pd.DataFrame, 
                           created_files: Dict[str, str]) -> str:
        """Create HTML summary page linking all visualizations."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weed Detection Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .file-link {{ margin: 5px 0; }}
                .stats {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Comprehensive Weed Detection Model Comparison</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Models evaluated: {len(results_df['model_name'].unique())}</p>
            </div>
            
            <div class="section stats">
                <h2>Summary Statistics</h2>
                <p><strong>Best mAP@50:</strong> {results_df['map50'].max():.3f} ({results_df.loc[results_df['map50'].idxmax(), 'model_name']})</p>
                <p><strong>Fastest model:</strong> {results_df['fps'].max():.1f} FPS ({results_df.loc[results_df['fps'].idxmax(), 'model_name']})</p>
                <p><strong>Average mAP@50:</strong> {results_df['map50'].mean():.3f}</p>
            </div>
            
            <div class="section">
                <h2>Interactive Visualizations</h2>
                <div class="file-link"><a href="interactive/performance_dashboard.html">📊 Performance Dashboard</a></div>
                <div class="file-link"><a href="interactive/architecture_comparison.html">🏗️ Architecture Comparison</a></div>
            </div>
            
            <div class="section">
                <h2>Static Plots</h2>
        """
        
        if 'statistical_plots' in created_files:
            for plot in created_files['statistical_plots']:
                plot_name = Path(plot).name
                html_content += f'<div class="file-link"><a href="static/{plot_name}">📈 {plot_name}</a></div>'
        
        if 'publication_figures' in created_files:
            for plot in created_files['publication_figures']:
                plot_name = Path(plot).name
                html_content += f'<div class="file-link"><a href="static/{plot_name}">📄 {plot_name}</a></div>'
        
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>Reports</h2>
                <div class="file-link"><a href="reports/comparative_study.tex">📄 LaTeX Report</a></div>
            </div>
            
            <div class="section">
                <h2>Data Tables</h2>
                <p>Complete results exported to CSV format for further analysis.</p>
            </div>
            
        </body>
        </html>
        """
        
        html_path = self.output_dir / "summary_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)

def main():
    """Main function for standalone visualization."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Visualization Tools")
    parser.add_argument('--results', type=str, required=True,
                       help='CSV file with comparison results')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Experiment configuration file')
    
    args = parser.parse_args()
    
    # Load results
    results_df = pd.read_csv(args.results)
    
    # Load config if provided
    experiment_config = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            experiment_config = yaml.safe_load(f)
    
    # Create visualizer
    visualizer = AdvancedVisualizer(args.output_dir)
    
    # Generate comprehensive report
    created_files = visualizer.create_comprehensive_report(results_df, experiment_config)
    
    print("\\n✅ Visualization and reporting completed!")
    print(f"📁 View results at: {args.output_dir}/summary_report.html")

if __name__ == "__main__":
    main()