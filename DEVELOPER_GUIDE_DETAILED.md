# Vision Flow Framework (VFF) - Developer's Complete Guide

## What is VFF?

The Vision Flow Framework is a comprehensive research platform designed for comparing multiple state-of-the-art object detection models specifically for agricultural weed detection. Think of it as a scientific benchmarking laboratory where researchers can fairly compare how well different AI models perform at identifying weeds in crop fields.

### Why Does This Matter?

Traditional weed detection in agriculture relies on manual inspection or broad herbicide application. This framework enables precision agriculture by:

- **Automated Detection**: AI models can identify weeds in real-time from drone or robot cameras
- **Reduced Chemical Use**: Targeted spraying only where weeds are detected
- **Cost Savings**: Less labor and herbicide costs for farmers
- **Research Foundation**: Provides standardized comparison of detection algorithms

## How the Framework Works

### The Big Picture Architecture

Imagine VFF as a sophisticated testing laboratory with four main components:

1. **The Orchestrator** (`master_framework.py`): Like a lab manager that coordinates all experiments
2. **The Training Factory** (`scripts/train.py`): Where different AI models learn to detect weeds
3. **The Configuration Manager** (`scripts/experiment_manager.py`): Keeps track of all experimental settings
4. **The Model Wrappers** (`scripts/models/`): Translators that make different AI architectures speak the same language

### Supported AI Models

The framework supports 9+ different object detection architectures:

**YOLO Family** (You Only Look Once):

- YOLOv8: Latest ultrafast detection with high accuracy
- YOLOv11: Newest version with improved efficiency
- YOLOv7: Previous generation with proven performance

**Specialized Architectures**:

- YOLO-NAS: Neural Architecture Search optimized YOLO
- YOLOX: Enhanced YOLO with anchor-free detection
- EfficientDet: Google's efficient detection model
- DETR: Facebook's transformer-based detector

### Datasets Supported

The framework works with specialized agricultural datasets:

- **DeepWeeds**: Australian weeds dataset with 17,509 images
- **Weed25**: 25 different weed species classification
- **CWD30**: Crop-Weed Detection with 30 classes
- **WeedsGalore**: Large-scale weed detection dataset

## Getting Started - Step by Step

### 1. Environment Setup

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

The framework uses conditional imports, meaning it gracefully handles missing optional dependencies. If you don't have YOLO-NAS installed, it simply won't be available for training.

### 2. Data Preparation

**Option A: Quick Testing with Dummy Data**

```bash
python scripts/generate_dummy_data.py --n_train 20 --n_val 10
```

This creates a small synthetic dataset for testing the pipeline.

**Option B: Download Real Agricultural Data**

```bash
python scripts/download_datasets.py --datasets deepweeds sample_weeds --sample 50
```

This downloads actual weed detection datasets (sampled to 50 images for quick testing).

### 3. Training Your First Model

**Simple Single Model Training:**

```bash
python scripts/train.py --data data/dummy.yaml --models yolov8n.pt --config configs/base.yaml
```

**Multi-Model Comparison:**

```bash
python scripts/train.py --data data/sample_weeds.yaml --models yolov8n.pt yolov8s.pt yolov11n.pt --config configs/base.yaml
```

**Full Research Study:**

```bash
python master_framework.py --experiment comprehensive_study --datasets all --models all
```

### 4. Evaluating Performance

**Basic Metrics (mAP, FPS, Latency):**

```bash
python scripts/evaluate.py --models_dir results/runs --device cuda
```

**Energy Consumption Analysis:**

```bash
python scripts/energy_logger.py --models results/runs/yolov8n/weights/best.pt --device cuda --n_images 100
```

**Robustness Testing:**

```bash
python scripts/perturb_eval.py --model results/runs/yolov8n/weights/best.pt --data data/sample_weeds.yaml
```

### 5. Model Optimization

**Pruning (Reduce Model Size):**

```bash
python scripts/prune.py --model results/runs/yolov8n/weights/best.pt --output models/yolov8n_pruned.pt --method unstructured --amount 0.3
```

**Quantization (Speed Up Inference):**

```bash
python scripts/quantize.py --model results/runs/yolov8n/weights/best.pt --formats onnx tensorrt --benchmark
```

## Understanding the Code Structure

### Core Design Patterns

**1. Factory Pattern (`ModelFactory`)**
Instead of hardcoding how to create each model type, the factory pattern lets you create any model with:

```python
model = ModelFactory.create_model("yolov8n.pt", num_classes=2)
```

This abstracts away the complexity of different model initialization procedures.

**2. Wrapper Pattern (Model Wrappers)**
Each AI architecture has different APIs. The wrapper pattern provides a unified interface:

```python
# All models support these same methods regardless of underlying architecture
results = model.train(data="data/weeds.yaml", epochs=50, batch_size=16)
metrics = model.validate(data="data/weeds.yaml", weights="best.pt")
predictions = model.predict(source="test_images/")
```

**3. Configuration-Driven Development**
Instead of hardcoding parameters, everything is controlled by YAML files:

- `configs/base.yaml`: Standard training parameters
- `configs/experiments/`: Complex multi-model studies
- `data/*.yaml`: Dataset configurations

### File Organization Logic

```
vff/
├── configs/                    # All experiment configurations
│   ├── base.yaml              # Standard hyperparameters
│   └── experiments/           # Complex study configurations
├── data/                      # Dataset configurations and data
│   ├── dummy.yaml             # Test dataset config
│   ├── sample_weeds.yaml      # Real dataset config
│   └── */                     # Actual image data folders
├── scripts/                   # All executable Python code
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Performance evaluation
│   ├── models/                # Model wrapper implementations
│   ├── download_datasets.py   # Data acquisition
│   └── ...                    # Various utilities
├── results/                   # Generated outputs
│   ├── runs/                  # Training results
│   ├── tables/               # Performance tables
│   ├── figures/              # Generated plots
│   └── *.json                # Summary statistics
└── models/                    # Optimized model files
```

### Critical Configuration Conventions

**Dataset YAML Structure:**

```yaml
# Must use absolute paths to prevent Ultralytics auto-relocation
path: c:/Users/smart/Downloads/vff/data/dummy
train: images/train
val: images/val
names:
  0: weed
  1: crop
```

**Training Configuration:**

```yaml
# configs/base.yaml
epochs: 1 # Set to 1 for testing, 50+ for real training
batch_size: 16 # Adjust based on GPU memory
device: cuda # or 'cpu' for CPU-only training
img_size: 640 # Standard YOLO input size
```

## Advanced Features

### Energy-Aware Training

The framework includes unique energy consumption monitoring:

```python
# Automatically logs power consumption during training
energy_logging:
  enabled: true
  interval_s: 1.0
```

### Robustness Testing

Test how models handle image perturbations (blur, noise, brightness changes):

```bash
python scripts/perturb_eval.py --model best.pt --data weeds.yaml
```

### Statistical Analysis

Generate publication-ready statistical comparisons:

```bash
python scripts/statistical_analysis.py --results_dir results/comprehensive_comparison/
```

### Experiment Management

For complex studies involving multiple models and datasets:

```python
# Create experiment configuration
config = ExperimentConfig(
    experiment_name="agricultural_optimization",
    models=["yolov8n.pt", "yolov11n.pt", "yolo_nas_s"],
    datasets=["deepweeds", "weed25"],
    epochs=100,
    description="Comparing efficiency vs accuracy trade-offs"
)
```

## Common Workflows

### Testing the Complete Pipeline

```bash
python test_project.py
```

This runs comprehensive tests to verify everything is working correctly.

### Adding a New Model Architecture

1. **Create the wrapper:**

```python
# scripts/models/my_new_model_wrapper.py
class MyNewModelWrapper:
    def train(self, data, epochs, batch_size, device, **kwargs):
        # Implementation specific to your model
        pass

    def validate(self, data, weights, device, **kwargs):
        # Validation logic
        pass

    def predict(self, source, weights, **kwargs):
        # Prediction logic
        pass
```

2. **Register in the factory:**

```python
# scripts/train.py
def create_model(model_name, num_classes):
    if model_name == "my_new_model":
        return MyNewModelWrapper(model_name, num_classes)
```

3. **Add conditional import:**

```python
try:
    from models.my_new_model_wrapper import MyNewModelWrapper
    MY_NEW_MODEL_AVAILABLE = True
except ImportError:
    MY_NEW_MODEL_AVAILABLE = False
```

### Debugging Common Issues

**Model not found:**
Check availability with:

```bash
python test_models_availability.py
```

**Path issues:**
Always use absolute paths in dataset YAML files to prevent Ultralytics from moving your data.

**Memory errors:**
Reduce batch_size in `configs/base.yaml` or switch device to 'cpu'.

## Results and Visualization

### Generated Outputs

After training, you'll find:

- `results/runs/`: Individual model training results
- `results/training_summary.json`: Aggregate training metrics
- `results/eval_summary.json`: Evaluation results across all models
- `results/figures/`: Publication-ready plots and charts

### Creating Publication Figures

```bash
python scripts/advanced_visualizer.py --results_dir results/comprehensive_comparison/
```

This generates professional plots comparing:

- Model accuracy (mAP scores)
- Inference speed (FPS)
- Energy consumption
- Model size vs performance trade-offs

## Research Applications

This framework is designed for:

- **Academic Research**: Comparing detection algorithms for agricultural applications
- **Industry Evaluation**: Selecting optimal models for deployment on farm equipment
- **Algorithm Development**: Benchmarking new detection architectures
- **Precision Agriculture**: Enabling targeted weed management systems

The standardized evaluation ensures fair comparisons and reproducible results across different research groups and applications.
