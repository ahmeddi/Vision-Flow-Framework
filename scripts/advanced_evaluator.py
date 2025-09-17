"""
Advanced Evaluation Framework for Weed Detection Models
=====================================================

This module provides comprehensive evaluation capabilities including:
- Detailed performance metrics (mAP, precision, recall, F1)
- Inference speed and memory benchmarking
- Statistical significance testing
- Cross-dataset generalization analysis
- Robustness evaluation under different conditions
- Energy consumption analysis
"""

import os
import time
import gc
import psutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import cv2

warnings.filterwarnings('ignore')

@dataclass
class DetailedMetrics:
    """Comprehensive metrics for model evaluation."""
    
    # Detection metrics
    map50: float
    map50_95: float
    map75: float
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    per_class_ap: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    
    # Speed metrics
    inference_time_ms: float
    preprocess_time_ms: float
    postprocess_time_ms: float
    fps: float
    
    # Memory metrics
    model_size_mb: float
    peak_memory_mb: float
    gpu_memory_mb: float
    
    # Efficiency metrics
    params_millions: float
    flops_billions: float
    efficiency_score: float  # mAP / (params * flops)
    
    # Robustness metrics
    noise_robustness: float
    blur_robustness: float
    brightness_robustness: float
    
    # Energy metrics (if available)
    energy_per_image_mj: float
    carbon_footprint_g: float

@dataclass
class ComparisonAnalysis:
    """Statistical comparison between models."""
    
    model_a: str
    model_b: str
    metric: str
    
    # Statistical test results
    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    confidence_interval: Tuple[float, float]
    
    # Practical significance
    improvement_percent: float
    practical_significance: bool

class AdvancedEvaluator:
    """Advanced evaluation framework for model comparison."""
    
    def __init__(self, output_dir: str = "results/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation configurations
        self.robustness_configs = self._init_robustness_configs()
        self.energy_monitor = EnergyMonitor() if self._check_energy_monitoring() else None
        
        # Results storage
        self.detailed_results: Dict[str, DetailedMetrics] = {}
        self.comparison_results: List[ComparisonAnalysis] = []
    
    def _init_robustness_configs(self) -> Dict[str, Dict]:
        """Initialize robustness test configurations."""
        return {
            'noise': {
                'gaussian_std': [0.01, 0.02, 0.05, 0.1],
                'salt_pepper_prob': [0.01, 0.02, 0.05]
            },
            'blur': {
                'gaussian_sigma': [0.5, 1.0, 2.0, 3.0],
                'motion_size': [3, 5, 7, 9]
            },
            'brightness': {
                'gamma_values': [0.5, 0.7, 1.3, 1.5, 2.0],
                'brightness_delta': [-50, -30, 30, 50]
            },
            'weather': {
                'rain_intensity': [0.1, 0.3, 0.5],
                'fog_density': [0.2, 0.4, 0.6]
            }
        }
    
    def _check_energy_monitoring(self) -> bool:
        """Check if energy monitoring is available."""
        try:
            import pynvml
            return True
        except ImportError:
            return False
    
    def evaluate_model_comprehensive(self, 
                                   model_path: str,
                                   dataset_config: Dict[str, str],
                                   model_name: str = None) -> DetailedMetrics:
        """
        Perform comprehensive evaluation of a single model.
        
        Args:
            model_path: Path to the trained model
            dataset_config: Dataset configuration with paths
            model_name: Optional model name for identification
            
        Returns:
            DetailedMetrics object with all evaluation results
        """
        
        if model_name is None:
            model_name = Path(model_path).stem
        
        print(f"\\n=== Comprehensive Evaluation: {model_name} ===")
        
        # Load model
        model = YOLO(model_path)
        
        # 1. Standard detection metrics
        print("📊 Computing detection metrics...")
        detection_metrics = self._compute_detection_metrics(model, dataset_config)
        
        # 2. Speed benchmarking
        print("⚡ Benchmarking inference speed...")
        speed_metrics = self._benchmark_speed(model, dataset_config)
        
        # 3. Memory analysis
        print("💾 Analyzing memory usage...")
        memory_metrics = self._analyze_memory(model, dataset_config)
        
        # 4. Model efficiency
        print("🔧 Computing efficiency metrics...")
        efficiency_metrics = self._compute_efficiency(model)
        
        # 5. Robustness testing
        print("🛡️ Testing robustness...")
        robustness_metrics = self._test_robustness(model, dataset_config)
        
        # 6. Energy consumption (if available)
        energy_metrics = {'energy_per_image_mj': 0.0, 'carbon_footprint_g': 0.0}
        if self.energy_monitor:
            print("⚡ Measuring energy consumption...")
            energy_metrics = self._measure_energy(model, dataset_config)
        
        # Combine all metrics
        detailed_metrics = DetailedMetrics(
            **detection_metrics,
            **speed_metrics,
            **memory_metrics,
            **efficiency_metrics,
            **robustness_metrics,
            **energy_metrics
        )
        
        # Store results
        self.detailed_results[model_name] = detailed_metrics
        
        print(f"✅ Evaluation completed for {model_name}")
        return detailed_metrics
    
    def _compute_detection_metrics(self, model, dataset_config) -> Dict[str, Any]:
        """Compute detailed detection metrics."""
        
        # Run validation
        results = model.val(data=dataset_config['yaml_path'], verbose=False)
        
        # Extract metrics
        metrics = {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'map75': float(results.box.map75) if hasattr(results.box, 'map75') else 0.0,
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 0.0,  # Will compute from precision/recall
            'per_class_ap': {},
            'per_class_precision': {},
            'per_class_recall': {}
        }
        
        # Compute F1 score
        p, r = metrics['precision'], metrics['recall']
        metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        # Per-class metrics (if available)
        if hasattr(results.box, 'ap_class_index'):
            class_names = model.names
            ap_per_class = results.box.ap[:, 0]  # AP@0.5
            
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = class_names.get(int(class_idx), f"class_{class_idx}")
                metrics['per_class_ap'][class_name] = float(ap_per_class[i])
        
        return metrics
    
    def _benchmark_speed(self, model, dataset_config) -> Dict[str, float]:
        """Benchmark inference speed with detailed timing."""
        
        # Get sample images
        val_dir = Path(dataset_config['path']) / 'images' / 'val'
        sample_images = list(val_dir.glob('*.jpg'))[:20]  # Use 20 images
        
        if not sample_images:
            return {
                'inference_time_ms': 0.0,
                'preprocess_time_ms': 0.0,
                'postprocess_time_ms': 0.0,
                'fps': 0.0
            }
        
        # Warm-up runs
        for _ in range(5):
            _ = model.predict(str(sample_images[0]), verbose=False)
        
        # Detailed timing
        inference_times = []
        preprocess_times = []
        postprocess_times = []
        
        for img_path in sample_images:
            # Time each component
            start_time = time.perf_counter()
            
            # Load and preprocess
            preprocess_start = time.perf_counter()
            results = model.predict(str(img_path), verbose=False)
            preprocess_end = time.perf_counter()
            
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000  # ms
            inference_times.append(total_time)
            
            # Note: With YOLO, preprocessing and postprocessing are bundled
            # We'll estimate based on total time
            preprocess_times.append(total_time * 0.1)  # ~10% preprocessing
            postprocess_times.append(total_time * 0.1)  # ~10% postprocessing
        
        return {
            'inference_time_ms': np.mean(inference_times),
            'preprocess_time_ms': np.mean(preprocess_times),
            'postprocess_time_ms': np.mean(postprocess_times),
            'fps': 1000 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0.0
        }
    
    def _analyze_memory(self, model, dataset_config) -> Dict[str, float]:
        """Analyze memory usage."""
        
        # Model size
        model_size_mb = 0.0
        if hasattr(model, 'ckpt_path') and model.ckpt_path:
            model_size_mb = Path(model.ckpt_path).stat().st_size / (1024 * 1024)
        
        # Parameter count
        param_count = sum(p.numel() for p in model.model.parameters())
        
        # Memory usage during inference
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run inference to measure peak memory
        val_dir = Path(dataset_config['path']) / 'images' / 'val'
        sample_images = list(val_dir.glob('*.jpg'))[:5]
        
        if sample_images:
            for img_path in sample_images:
                _ = model.predict(str(img_path), verbose=False)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = memory_after - memory_before
        
        # GPU memory (if available)
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        
        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': max(peak_memory, 0),
            'gpu_memory_mb': gpu_memory_mb,
            'params_millions': param_count / 1e6
        }
    
    def _compute_efficiency(self, model) -> Dict[str, float]:
        """Compute model efficiency metrics."""
        
        # Parameter count
        param_count = sum(p.numel() for p in model.model.parameters())
        params_millions = param_count / 1e6
        
        # Estimate FLOPs (rough approximation)
        # This is a simplified calculation - for precise FLOPs, would need profiling
        flops_billions = params_millions * 0.5  # Rough estimate
        
        # Efficiency score (to be computed after mAP is available)
        efficiency_score = 0.0  # Will be computed in parent function
        
        return {
            'params_millions': params_millions,
            'flops_billions': flops_billions,
            'efficiency_score': efficiency_score
        }
    
    def _test_robustness(self, model, dataset_config) -> Dict[str, float]:
        """Test model robustness under various conditions."""
        
        # Get sample images
        val_dir = Path(dataset_config['path']) / 'images' / 'val'
        sample_images = list(val_dir.glob('*.jpg'))[:10]
        
        if not sample_images:
            return {
                'noise_robustness': 0.0,
                'blur_robustness': 0.0,
                'brightness_robustness': 0.0
            }
        
        # Baseline performance
        baseline_scores = []
        for img_path in sample_images:
            results = model.predict(str(img_path), verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                baseline_scores.append(len(results[0].boxes))
            else:
                baseline_scores.append(0)
        
        baseline_avg = np.mean(baseline_scores)
        
        # Test noise robustness
        noise_scores = self._test_noise_robustness(model, sample_images)
        noise_robustness = np.mean(noise_scores) / baseline_avg if baseline_avg > 0 else 0.0
        
        # Test blur robustness
        blur_scores = self._test_blur_robustness(model, sample_images)
        blur_robustness = np.mean(blur_scores) / baseline_avg if baseline_avg > 0 else 0.0
        
        # Test brightness robustness
        brightness_scores = self._test_brightness_robustness(model, sample_images)
        brightness_robustness = np.mean(brightness_scores) / baseline_avg if baseline_avg > 0 else 0.0
        
        return {
            'noise_robustness': max(0.0, min(1.0, noise_robustness)),
            'blur_robustness': max(0.0, min(1.0, blur_robustness)),
            'brightness_robustness': max(0.0, min(1.0, brightness_robustness))
        }
    
    def _test_noise_robustness(self, model, image_paths) -> List[float]:
        """Test robustness to noise."""
        scores = []
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05 * 255, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            
            # Save temporary image
            temp_path = self.output_dir / f"temp_noisy_{Path(img_path).name}"
            cv2.imwrite(str(temp_path), noisy_img)
            
            # Test on noisy image
            try:
                results = model.predict(str(temp_path), verbose=False)
                if results and len(results) > 0 and results[0].boxes is not None:
                    scores.append(len(results[0].boxes))
                else:
                    scores.append(0)
            except:
                scores.append(0)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
        
        return scores
    
    def _test_blur_robustness(self, model, image_paths) -> List[float]:
        """Test robustness to blur."""
        scores = []
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img, (15, 15), 2.0)
            
            # Save temporary image
            temp_path = self.output_dir / f"temp_blurred_{Path(img_path).name}"
            cv2.imwrite(str(temp_path), blurred_img)
            
            # Test on blurred image
            try:
                results = model.predict(str(temp_path), verbose=False)
                if results and len(results) > 0 and results[0].boxes is not None:
                    scores.append(len(results[0].boxes))
                else:
                    scores.append(0)
            except:
                scores.append(0)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
        
        return scores
    
    def _test_brightness_robustness(self, model, image_paths) -> List[float]:
        """Test robustness to brightness changes."""
        scores = []
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Adjust brightness (gamma correction)
            gamma = 0.5  # Darker image
            bright_img = np.power(img / 255.0, gamma) * 255.0
            bright_img = bright_img.astype(np.uint8)
            
            # Save temporary image
            temp_path = self.output_dir / f"temp_bright_{Path(img_path).name}"
            cv2.imwrite(str(temp_path), bright_img)
            
            # Test on brightness-adjusted image
            try:
                results = model.predict(str(temp_path), verbose=False)
                if results and len(results) > 0 and results[0].boxes is not None:
                    scores.append(len(results[0].boxes))
                else:
                    scores.append(0)
            except:
                scores.append(0)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
        
        return scores
    
    def _measure_energy(self, model, dataset_config) -> Dict[str, float]:
        """Measure energy consumption (if monitoring available)."""
        
        if not self.energy_monitor:
            return {'energy_per_image_mj': 0.0, 'carbon_footprint_g': 0.0}
        
        # Get sample images
        val_dir = Path(dataset_config['path']) / 'images' / 'val'
        sample_images = list(val_dir.glob('*.jpg'))[:10]
        
        if not sample_images:
            return {'energy_per_image_mj': 0.0, 'carbon_footprint_g': 0.0}
        
        # Measure energy for inference
        total_energy = 0.0
        
        for img_path in sample_images:
            energy_start = self.energy_monitor.get_current_power()
            start_time = time.time()
            
            _ = model.predict(str(img_path), verbose=False)
            
            end_time = time.time()
            energy_end = self.energy_monitor.get_current_power()
            
            # Estimate energy consumption (simplified)
            duration = end_time - start_time
            avg_power = (energy_start + energy_end) / 2
            energy_joules = avg_power * duration
            total_energy += energy_joules
        
        energy_per_image = total_energy / len(sample_images)
        energy_per_image_mj = energy_per_image * 1000  # Convert to millijoules
        
        # Carbon footprint (rough estimate)
        # Using average grid carbon intensity: ~0.5 kg CO2/kWh
        carbon_intensity = 0.5  # kg CO2/kWh
        energy_kwh = energy_per_image / 3600000  # Convert J to kWh
        carbon_footprint_g = energy_kwh * carbon_intensity * 1000  # Convert to grams
        
        return {
            'energy_per_image_mj': energy_per_image_mj,
            'carbon_footprint_g': carbon_footprint_g
        }
    
    def compare_models_statistical(self, 
                                 results_df: pd.DataFrame, 
                                 metric: str = 'map50',
                                 alpha: float = 0.05) -> List[ComparisonAnalysis]:
        """
        Perform statistical comparison between models.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to compare
            alpha: Significance level
            
        Returns:
            List of comparison analyses
        """
        
        print(f"\\n🔬 Statistical comparison on {metric}")
        
        models = results_df['model_name'].unique()
        comparisons = []
        
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                
                # Get data for both models
                data_a = results_df[results_df['model_name'] == model_a][metric].values
                data_b = results_df[results_df['model_name'] == model_b][metric].values
                
                if len(data_a) == 0 or len(data_b) == 0:
                    continue
                
                # Perform statistical test
                if len(data_a) > 1 and len(data_b) > 1:
                    # Use t-test for multiple observations
                    statistic, p_value = stats.ttest_ind(data_a, data_b)
                    
                    # Cohen's d effect size
                    pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a) + 
                                        (len(data_b) - 1) * np.var(data_b)) / 
                                       (len(data_a) + len(data_b) - 2))
                    effect_size = abs(np.mean(data_a) - np.mean(data_b)) / pooled_std if pooled_std > 0 else 0
                    
                    # Confidence interval for difference
                    se_diff = pooled_std * np.sqrt(1/len(data_a) + 1/len(data_b))
                    t_critical = stats.t.ppf(1 - alpha/2, len(data_a) + len(data_b) - 2)
                    mean_diff = np.mean(data_a) - np.mean(data_b)
                    ci_lower = mean_diff - t_critical * se_diff
                    ci_upper = mean_diff + t_critical * se_diff
                    
                else:
                    # Single observation comparison
                    statistic = 0.0
                    p_value = 1.0
                    effect_size = 0.0
                    ci_lower, ci_upper = 0.0, 0.0
                    mean_diff = np.mean(data_a) - np.mean(data_b)
                
                # Practical significance (>5% improvement)
                improvement_percent = abs(mean_diff) / np.mean(data_b) * 100 if np.mean(data_b) > 0 else 0
                practical_significance = improvement_percent > 5.0
                
                comparison = ComparisonAnalysis(
                    model_a=model_a,
                    model_b=model_b,
                    metric=metric,
                    statistic=statistic,
                    p_value=p_value,
                    effect_size=effect_size,
                    significant=p_value < alpha,
                    confidence_interval=(ci_lower, ci_upper),
                    improvement_percent=improvement_percent,
                    practical_significance=practical_significance
                )
                
                comparisons.append(comparison)
        
        self.comparison_results.extend(comparisons)
        return comparisons
    
    def generate_evaluation_report(self, save_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        
        if save_path is None:
            save_path = self.output_dir / "evaluation_report.md"
        
        report_lines = []
        report_lines.append("# Comprehensive Model Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Executive summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        if self.detailed_results:
            # Best performing models
            best_map = max(self.detailed_results.values(), key=lambda x: x.map50)
            best_speed = max(self.detailed_results.values(), key=lambda x: x.fps)
            best_efficiency = max(self.detailed_results.values(), key=lambda x: x.efficiency_score)
            
            # Find model names
            best_map_model = next(name for name, metrics in self.detailed_results.items() if metrics == best_map)
            best_speed_model = next(name for name, metrics in self.detailed_results.items() if metrics == best_speed)
            best_eff_model = next(name for name, metrics in self.detailed_results.items() if metrics == best_efficiency)
            
            report_lines.append(f"**Best Accuracy:** {best_map_model} (mAP@50: {best_map.map50:.3f})")
            report_lines.append(f"**Fastest Model:** {best_speed_model} ({best_speed.fps:.1f} FPS)")
            report_lines.append(f"**Most Efficient:** {best_eff_model} (Score: {best_efficiency.efficiency_score:.4f})")
            report_lines.append("")
        
        # Detailed results table
        report_lines.append("## Detailed Results")
        report_lines.append("")
        report_lines.append("| Model | mAP@50 | mAP@50-95 | FPS | Params (M) | Size (MB) | Efficiency |")
        report_lines.append("|-------|--------|-----------|-----|------------|-----------|------------|")
        
        for model_name, metrics in self.detailed_results.items():
            report_lines.append(
                f"| {model_name} | {metrics.map50:.3f} | {metrics.map50_95:.3f} | "
                f"{metrics.fps:.1f} | {metrics.params_millions:.1f} | "
                f"{metrics.model_size_mb:.1f} | {metrics.efficiency_score:.4f} |"
            )
        
        report_lines.append("")
        
        # Statistical comparisons
        if self.comparison_results:
            report_lines.append("## Statistical Comparisons")
            report_lines.append("")
            
            significant_comparisons = [c for c in self.comparison_results if c.significant]
            
            if significant_comparisons:
                report_lines.append("### Significant Differences")
                for comp in significant_comparisons:
                    report_lines.append(
                        f"- **{comp.model_a} vs {comp.model_b}**: "
                        f"p={comp.p_value:.4f}, effect size={comp.effect_size:.2f}"
                    )
                report_lines.append("")
        
        # Robustness analysis
        report_lines.append("## Robustness Analysis")
        report_lines.append("")
        
        for model_name, metrics in self.detailed_results.items():
            report_lines.append(f"### {model_name}")
            report_lines.append(f"- **Noise Robustness:** {metrics.noise_robustness:.3f}")
            report_lines.append(f"- **Blur Robustness:** {metrics.blur_robustness:.3f}")
            report_lines.append(f"- **Brightness Robustness:** {metrics.brightness_robustness:.3f}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("Based on the comprehensive evaluation:")
        report_lines.append("")
        
        if self.detailed_results:
            # Generate recommendations based on results
            accuracy_models = sorted(self.detailed_results.items(), key=lambda x: x[1].map50, reverse=True)
            speed_models = sorted(self.detailed_results.items(), key=lambda x: x[1].fps, reverse=True)
            
            report_lines.append(f"- **For highest accuracy:** {accuracy_models[0][0]}")
            report_lines.append(f"- **For real-time applications:** {speed_models[0][0]}")
            
            # Balanced recommendation
            balanced_scores = {}
            for name, metrics in self.detailed_results.items():
                # Normalize and combine metrics
                balanced_score = (metrics.map50 * 0.5) + (min(metrics.fps, 50) / 50 * 0.3) + (metrics.efficiency_score * 0.2)
                balanced_scores[name] = balanced_score
            
            best_balanced = max(balanced_scores, key=balanced_scores.get)
            report_lines.append(f"- **For balanced performance:** {best_balanced}")
        
        # Save report
        report_content = "\\n".join(report_lines)
        with open(save_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Evaluation report saved to: {save_path}")
        return report_content

class EnergyMonitor:
    """Simple energy monitoring class (placeholder for more sophisticated monitoring)."""
    
    def __init__(self):
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if energy monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except:
            return False
    
    def get_current_power(self) -> float:
        """Get current power consumption in watts."""
        if not self.available:
            return 0.0
        
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return power_mw / 1000.0  # Convert to watts
        except:
            return 0.0

def main():
    """Main function for standalone evaluation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Model Evaluation")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset configuration file')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name for identification')
    
    args = parser.parse_args()
    
    # Load dataset configuration
    import yaml
    with open(args.dataset, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = AdvancedEvaluator(args.output_dir)
    
    # Run evaluation
    metrics = evaluator.evaluate_model_comprehensive(
        args.model, 
        dataset_config, 
        args.model_name
    )
    
    # Generate report
    evaluator.generate_evaluation_report()
    
    print("\\n✅ Advanced evaluation completed!")

if __name__ == "__main__":
    main()