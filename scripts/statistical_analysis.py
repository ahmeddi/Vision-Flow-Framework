"""Statistical analysis module for model comparison in weed detection benchmark."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, spearmanr
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for model benchmarking."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def bootstrap_confidence_interval(self, 
                                    data: np.ndarray, 
                                    n_bootstrap: int = 2000,
                                    confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.
        
        Args:
            data: Array of metric values (e.g., mAP scores per image)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = resample(data, replace=True, n_samples=len(data))
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        mean_estimate = np.mean(bootstrap_means)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return mean_estimate, lower_bound, upper_bound
    
    def wilcoxon_comparison(self, 
                          model_a_results: np.ndarray,
                          model_b_results: np.ndarray,
                          alternative: str = 'two-sided') -> Dict:
        """
        Perform Wilcoxon signed-rank test to compare two models.
        
        Args:
            model_a_results: Results from model A (per image/sample)
            model_b_results: Results from model B (per image/sample)
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        if len(model_a_results) != len(model_b_results):
            raise ValueError("Both models must have same number of results")
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(model_a_results, model_b_results, 
                                     alternative=alternative)
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(model_a_results)
        z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
        effect_size = abs(z_score) / np.sqrt(n)
        
        # Interpret effect size
        if effect_size < 0.1:
            effect_interpretation = "negligible"
        elif effect_size < 0.3:
            effect_interpretation = "small"
        elif effect_size < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'effect_interpretation': effect_interpretation,
            'mean_diff': np.mean(model_a_results) - np.mean(model_b_results),
            'median_diff': np.median(model_a_results) - np.median(model_b_results)
        }
    
    def multiple_comparisons_correction(self, 
                                      p_values: List[float], 
                                      method: str = 'holm') -> List[float]:
        """
        Apply multiple comparisons correction.
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            List of corrected p-values
        """
        p_values = np.array(p_values)
        n_comparisons = len(p_values)
        
        if method == 'bonferroni':
            return np.minimum(p_values * n_comparisons, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (n_comparisons - i), 1.0)
                
            return corrected
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)
            
            for i, idx in enumerate(reversed(sorted_indices)):
                rank = n_comparisons - i
                corrected[idx] = min(p_values[idx] * n_comparisons / rank, 1.0)
                
            return corrected
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def spearman_correlation_matrix(self, 
                                   metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate Spearman correlation matrix between metrics.
        
        Args:
            metrics_df: DataFrame with metrics as columns
            
        Returns:
            Tuple of (correlation_matrix, p_value_matrix)
        """
        metrics = metrics_df.select_dtypes(include=[np.number])
        n_metrics = len(metrics.columns)
        
        corr_matrix = np.zeros((n_metrics, n_metrics))
        p_value_matrix = np.zeros((n_metrics, n_metrics))
        
        for i, col1 in enumerate(metrics.columns):
            for j, col2 in enumerate(metrics.columns):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                else:
                    corr, p_val = spearmanr(metrics[col1], metrics[col2])
                    corr_matrix[i, j] = corr
                    p_value_matrix[i, j] = p_val
        
        corr_df = pd.DataFrame(corr_matrix, 
                              index=metrics.columns, 
                              columns=metrics.columns)
        p_value_df = pd.DataFrame(p_value_matrix, 
                                 index=metrics.columns, 
                                 columns=metrics.columns)
        
        return corr_df, p_value_df
    
    def pairwise_model_comparison(self, 
                                results_dict: Dict[str, np.ndarray],
                                metric_name: str = "mAP") -> pd.DataFrame:
        """
        Perform pairwise comparison between all models.
        
        Args:
            results_dict: Dictionary mapping model names to result arrays
            metric_name: Name of the metric being compared
            
        Returns:
            DataFrame with comparison results
        """
        model_names = list(results_dict.keys())
        n_models = len(model_names)
        
        comparisons = []
        p_values = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model_a = model_names[i]
                model_b = model_names[j]
                
                comparison_result = self.wilcoxon_comparison(
                    results_dict[model_a], 
                    results_dict[model_b]
                )
                
                comparisons.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'metric': metric_name,
                    'mean_a': np.mean(results_dict[model_a]),
                    'mean_b': np.mean(results_dict[model_b]),
                    'mean_diff': comparison_result['mean_diff'],
                    'p_value': comparison_result['p_value'],
                    'effect_size': comparison_result['effect_size'],
                    'significant': comparison_result['significant']
                })
                
                p_values.append(comparison_result['p_value'])
        
        # Apply multiple comparisons correction
        corrected_p_values = self.multiple_comparisons_correction(p_values)
        
        # Update significance based on corrected p-values
        for i, comp in enumerate(comparisons):
            comp['p_value_corrected'] = corrected_p_values[i]
            comp['significant_corrected'] = corrected_p_values[i] < self.alpha
        
        return pd.DataFrame(comparisons)
    
    def generate_statistical_report(self, 
                                  results_dict: Dict[str, Dict],
                                  output_path: str):
        """
        Generate comprehensive statistical analysis report.
        
        Args:
            results_dict: Nested dictionary with model results
            output_path: Path to save the report
        """
        report = {
            'summary_statistics': {},
            'confidence_intervals': {},
            'pairwise_comparisons': {},
            'correlation_analysis': {}
        }
        
        # Extract metrics for analysis
        metrics = ['mAP_50', 'mAP_50_95', 'fps', 'latency_ms', 'energy_j_per_frame']
        
        for metric in metrics:
            if metric in results_dict[list(results_dict.keys())[0]]:
                metric_data = {model: results_dict[model][metric] 
                              for model in results_dict.keys()}
                
                # Calculate confidence intervals
                report['confidence_intervals'][metric] = {}
                for model, data in metric_data.items():
                    if isinstance(data, (list, np.ndarray)):
                        mean, lower, upper = self.bootstrap_confidence_interval(np.array(data))
                        report['confidence_intervals'][metric][model] = {
                            'mean': float(mean),
                            'ci_lower': float(lower),
                            'ci_upper': float(upper)
                        }
                
                # Pairwise comparisons
                if all(isinstance(data, (list, np.ndarray)) for data in metric_data.values()):
                    comparison_df = self.pairwise_model_comparison(metric_data, metric)
                    report['pairwise_comparisons'][metric] = comparison_df.to_dict('records')
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Statistical analysis report saved to {output_path}")
        
        return report

def plot_comparison_heatmap(comparison_df: pd.DataFrame, 
                          metric: str, 
                          save_path: Optional[str] = None):
    """Plot heatmap of pairwise comparison results."""
    models = list(set(comparison_df['model_a'].tolist() + comparison_df['model_b'].tolist()))
    n_models = len(models)
    
    # Create matrices for visualization
    p_value_matrix = np.ones((n_models, n_models))
    effect_size_matrix = np.zeros((n_models, n_models))
    
    model_to_idx = {model: i for i, model in enumerate(models)}
    
    for _, row in comparison_df.iterrows():
        i, j = model_to_idx[row['model_a']], model_to_idx[row['model_b']]
        p_value_matrix[i, j] = row['p_value_corrected']
        p_value_matrix[j, i] = row['p_value_corrected']
        effect_size_matrix[i, j] = row['effect_size']
        effect_size_matrix[j, i] = row['effect_size']
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # P-value heatmap
    sns.heatmap(-np.log10(p_value_matrix), 
                xticklabels=models, yticklabels=models,
                annot=True, fmt='.2f', cmap='viridis',
                ax=ax1, cbar_kws={'label': '-log10(p-value)'})
    ax1.set_title(f'Statistical Significance\n{metric}')
    
    # Effect size heatmap
    sns.heatmap(effect_size_matrix, 
                xticklabels=models, yticklabels=models,
                annot=True, fmt='.3f', cmap='plasma',
                ax=ax2, cbar_kws={'label': 'Effect Size (r)'})
    ax2.set_title(f'Effect Size\n{metric}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison heatmap saved to {save_path}")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    analyzer = StatisticalAnalyzer()
    
    # Simulate model results
    models_data = {
        'yolov8n': np.random.normal(0.75, 0.05, 100),
        'yolov8s': np.random.normal(0.78, 0.04, 100),
        'yolov11n': np.random.normal(0.76, 0.04, 100),
        'yolo_nas_s': np.random.normal(0.79, 0.045, 100),
    }
    
    # Perform pairwise comparison
    comparison_results = analyzer.pairwise_model_comparison(models_data, "mAP@0.5")
    print("Pairwise Comparison Results:")
    print(comparison_results[['model_a', 'model_b', 'mean_diff', 
                            'p_value', 'p_value_corrected', 'significant_corrected']])
    
    # Plot comparison heatmap
    plot_comparison_heatmap(comparison_results, "mAP@0.5")
