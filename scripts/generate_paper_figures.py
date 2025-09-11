"""Generate figures and tables for research paper."""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Any, List
from statistical_analysis import StatisticalAnalyzer

# Set style for paper-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PaperFigureGenerator:
    """Generate publication-ready figures and tables."""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper formatting
        self.figsize = (10, 6)
        self.dpi = 300
        self.font_size = 12
        
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
        })
    
    def load_benchmark_data(self, results_file: str) -> pd.DataFrame:
        """Load and process benchmark results into DataFrame."""
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        rows = []
        
        # Extract training results
        if 'training' in data:
            for dataset, models in data['training'].items():
                for model, result in models.items():
                    if result['status'] == 'success':
                        eval_data = result.get('evaluation', {})
                        
                        row = {
                            'dataset': dataset,
                            'model': model,
                            'architecture': eval_data.get('model_type', 'Unknown'),
                            'map50': eval_data.get('map50', 0),
                            'map50_95': eval_data.get('map50_95', 0),
                            'precision': eval_data.get('precision', 0),
                            'recall': eval_data.get('recall', 0),
                            'fps': eval_data.get('fps', 0),
                            'latency_ms': eval_data.get('latency_ms', 0),
                            'parameters': eval_data.get('total_parameters', 0),
                            'model_size_mb': eval_data.get('model_size_mb', 0),
                            'training_time': result.get('training_time', 0)
                        }
                        
                        # Add energy data if available
                        if 'energy' in data:
                            energy_data = data['energy'].get(model, {})
                            row['energy_j_per_frame'] = energy_data.get('energy_j_per_frame', 0)
                            row['power_w'] = energy_data.get('average_power_w', 0)
                        
                        rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_performance_comparison_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table comparing model performance."""
        
        # Group by model and calculate means across datasets
        summary = df.groupby(['model', 'architecture']).agg({
            'map50': ['mean', 'std'],
            'map50_95': ['mean', 'std'],
            'fps': ['mean', 'std'], 
            'latency_ms': ['mean', 'std'],
            'parameters': 'mean',
            'model_size_mb': 'mean'
        }).round(3)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Create LaTeX table
        latex_table = "\\begin{table}[htbp]\n\\centering\n"
        latex_table += "\\caption{Performance comparison of object detection models for weed detection}\n"
        latex_table += "\\label{tab:performance_comparison}\n"
        latex_table += "\\begin{tabular}{l|l|c|c|c|c|c|c}\n"
        latex_table += "\\hline\n"
        latex_table += "Model & Architecture & mAP@0.5 & mAP@0.5:0.95 & FPS & Latency (ms) & Params (M) & Size (MB) \\\\\n"
        latex_table += "\\hline\n"
        
        for _, row in summary.iterrows():
            model = row['model'].replace('_', '\\_').replace('.pt', '')
            arch = row['architecture']
            map50 = f"{row['map50_mean']:.3f} ± {row['map50_std']:.3f}"
            map50_95 = f"{row['map50_95_mean']:.3f} ± {row['map50_95_std']:.3f}"
            fps = f"{row['fps_mean']:.1f} ± {row['fps_std']:.1f}"
            latency = f"{row['latency_ms_mean']:.1f} ± {row['latency_ms_std']:.1f}"
            params = f"{row['parameters_mean']/1e6:.1f}"
            size = f"{row['model_size_mb_mean']:.1f}"
            
            latex_table += f"{model} & {arch} & {map50} & {map50_95} & {fps} & {latency} & {params} & {size} \\\\\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        # Save table
        table_file = self.output_dir / "performance_comparison_table.tex"
        with open(table_file, 'w') as f:
            f.write(latex_table)
        
        print(f"Performance comparison table saved to {table_file}")
        return latex_table
    
    def plot_accuracy_vs_speed(self, df: pd.DataFrame):
        """Generate accuracy vs speed scatter plot (Figure for paper)."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color by architecture
        architectures = df['architecture'].unique()
        colors = sns.color_palette("Set2", len(architectures))
        arch_color_map = dict(zip(architectures, colors))
        
        for arch in architectures:
            arch_data = df[df['architecture'] == arch]
            ax.scatter(arch_data['fps'], arch_data['map50'], 
                      c=[arch_color_map[arch]], 
                      label=arch, 
                      alpha=0.7, 
                      s=100,
                      edgecolors='black',
                      linewidth=0.5)
        
        # Add model labels for key points
        for _, row in df.iterrows():
            if row['map50'] > 0.7 or row['fps'] > 100:  # Label high-performance models
                ax.annotate(row['model'].replace('.pt', ''), 
                           (row['fps'], row['map50']),
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9,
                           alpha=0.8)
        
        ax.set_xlabel('FPS (Frames Per Second)')
        ax.set_ylabel('mAP@0.5')
        ax.set_title('Accuracy vs Speed Trade-off for Weed Detection Models')
        ax.legend(title='Architecture', loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add Pareto frontier
        pareto_points = []
        for i, row1 in df.iterrows():
            is_pareto = True
            for j, row2 in df.iterrows():
                if i != j and row2['fps'] >= row1['fps'] and row2['map50'] >= row1['map50']:
                    if row2['fps'] > row1['fps'] or row2['map50'] > row1['map50']:
                        is_pareto = False
                        break
            if is_pareto:
                pareto_points.append((row1['fps'], row1['map50']))
        
        if pareto_points:
            pareto_points.sort()
            pareto_fps, pareto_map = zip(*pareto_points)
            ax.plot(pareto_fps, pareto_map, 'r--', alpha=0.7, linewidth=2, label='Pareto Frontier')
        
        plt.tight_layout()
        fig_file = self.output_dir / "accuracy_vs_speed.png"
        plt.savefig(fig_file, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(str(fig_file).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Accuracy vs speed plot saved to {fig_file}")
        plt.show()
    
    def plot_model_efficiency(self, df: pd.DataFrame):
        """Generate model efficiency comparison (size vs accuracy)."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Model size vs accuracy
        for arch in df['architecture'].unique():
            arch_data = df[df['architecture'] == arch]
            ax1.scatter(arch_data['model_size_mb'], arch_data['map50'], 
                       label=arch, alpha=0.7, s=100)
        
        ax1.set_xlabel('Model Size (MB)')
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('Model Size vs Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameters vs accuracy
        for arch in df['architecture'].unique():
            arch_data = df[df['architecture'] == arch]
            ax2.scatter(arch_data['parameters']/1e6, arch_data['map50'], 
                       label=arch, alpha=0.7, s=100)
        
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_ylabel('mAP@0.5')
        ax2.set_title('Model Parameters vs Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_file = self.output_dir / "model_efficiency.png"
        plt.savefig(fig_file, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(str(fig_file).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Model efficiency plot saved to {fig_file}")
        plt.show()
    
    def plot_energy_analysis(self, df: pd.DataFrame):
        """Generate energy consumption analysis."""
        
        # Filter models with energy data
        energy_df = df[df['energy_j_per_frame'] > 0].copy()
        
        if energy_df.empty:
            print("No energy data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Energy vs Accuracy
        for arch in energy_df['architecture'].unique():
            arch_data = energy_df[energy_df['architecture'] == arch]
            ax1.scatter(arch_data['energy_j_per_frame'], arch_data['map50'], 
                       label=arch, alpha=0.7, s=100)
        
        ax1.set_xlabel('Energy per Frame (Joules)')
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('Energy Consumption vs Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy efficiency (mAP per Joule)
        energy_df['energy_efficiency'] = energy_df['map50'] / energy_df['energy_j_per_frame']
        
        energy_summary = energy_df.groupby('model')['energy_efficiency'].mean().sort_values(ascending=False)
        
        ax2.bar(range(len(energy_summary)), energy_summary.values)
        ax2.set_xticks(range(len(energy_summary)))
        ax2.set_xticklabels([m.replace('.pt', '') for m in energy_summary.index], rotation=45)
        ax2.set_ylabel('Energy Efficiency (mAP/Joule)')
        ax2.set_title('Model Energy Efficiency Ranking')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_file = self.output_dir / "energy_analysis.png"
        plt.savefig(fig_file, dpi=self.dpi, bbox_inches='tight')
        print(f"Energy analysis plot saved to {fig_file}")
        plt.show()
    
    def plot_dataset_comparison(self, df: pd.DataFrame):
        """Compare model performance across datasets."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        metrics = ['map50', 'map50_95', 'fps', 'latency_ms']
        metric_titles = ['mAP@0.5', 'mAP@0.5:0.95', 'FPS', 'Latency (ms)']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            pivot_data = df.pivot_table(index='model', columns='dataset', values=metric, aggfunc='mean')
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', 
                       cmap='viridis', ax=axes[i], cbar_kws={'shrink': .8})
            axes[i].set_title(f'{title} by Model and Dataset')
            axes[i].set_xlabel('Dataset')
            axes[i].set_ylabel('Model')
        
        plt.tight_layout()
        fig_file = self.output_dir / "dataset_comparison.png"
        plt.savefig(fig_file, dpi=self.dpi, bbox_inches='tight')
        print(f"Dataset comparison plot saved to {fig_file}")
        plt.show()
    
    def plot_architecture_comparison(self, df: pd.DataFrame):
        """Compare different model architectures."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plots comparing architectures
        metrics = ['map50', 'fps', 'model_size_mb', 'parameters']
        metric_labels = ['mAP@0.5', 'FPS', 'Model Size (MB)', 'Parameters (M)']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i//2, i%2]
            
            data_for_plot = df.copy()
            if metric == 'parameters':
                data_for_plot[metric] = data_for_plot[metric] / 1e6
            
            sns.boxplot(data=data_for_plot, x='architecture', y=metric, ax=ax)
            ax.set_title(f'{label} by Architecture')
            ax.set_xlabel('Architecture')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        fig_file = self.output_dir / "architecture_comparison.png"
        plt.savefig(fig_file, dpi=self.dpi, bbox_inches='tight')
        print(f"Architecture comparison plot saved to {fig_file}")
        plt.show()
    
    def generate_statistical_analysis(self, df: pd.DataFrame):
        """Generate statistical analysis and comparison tables."""
        
        analyzer = StatisticalAnalyzer()
        
        # Prepare data for statistical analysis
        models_data = {}
        for model in df['model'].unique():
            model_data = df[df['model'] == model]['map50'].values
            if len(model_data) > 0:
                models_data[model] = model_data
        
        # Pairwise comparisons
        comparison_results = analyzer.pairwise_model_comparison(models_data, "mAP@0.5")
        
        # Save comparison results
        comparison_file = self.output_dir / "statistical_comparisons.csv"
        comparison_results.to_csv(comparison_file, index=False)
        print(f"Statistical comparisons saved to {comparison_file}")
        
        # Generate comparison heatmap
        from statistical_analysis import plot_comparison_heatmap
        plot_comparison_heatmap(comparison_results, "mAP@0.5", 
                               str(self.output_dir / "statistical_heatmap.png"))
        
        return comparison_results
    
    def generate_all_figures(self, results_file: str):
        """Generate all figures and tables for the paper."""
        
        print("Loading benchmark data...")
        df = self.load_benchmark_data(results_file)
        
        if df.empty:
            print("No data available for figure generation")
            return
        
        print(f"Loaded data for {len(df)} model-dataset combinations")
        
        print("Generating figures and tables...")
        
        # Main performance figures
        self.plot_accuracy_vs_speed(df)
        self.plot_model_efficiency(df) 
        self.plot_architecture_comparison(df)
        
        # Dataset analysis
        if len(df['dataset'].unique()) > 1:
            self.plot_dataset_comparison(df)
        
        # Energy analysis (if data available)
        if 'energy_j_per_frame' in df.columns:
            self.plot_energy_analysis(df)
        
        # Generate tables
        latex_table = self.generate_performance_comparison_table(df)
        
        # Statistical analysis
        self.generate_statistical_analysis(df)
        
        print(f"\nAll figures and tables generated in: {self.output_dir}")
        print("Files generated:")
        for file in self.output_dir.iterdir():
            print(f"  - {file.name}")

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures and tables')
    parser.add_argument('--results', required=True,
                       help='Path to benchmark results JSON file')
    parser.add_argument('--output_dir', default='results/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"Results file not found: {args.results}")
        return
    
    generator = PaperFigureGenerator(args.output_dir)
    generator.generate_all_figures(args.results)

if __name__ == '__main__':
    main()
