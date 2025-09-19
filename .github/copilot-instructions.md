# Vision Flow Framework (VFF) - AI Coding Instructions

This is a comprehensive weed detection research framework comparing multiple object detection architectures (YOLOv8, YOLOv11, YOLO-NAS, YOLOX, EfficientDet, DETR) across specialized agricultural datasets.

## Architecture Overview

**Core Components:**

- `master_framework.py` - Main orchestration script with `WeedDetectionMasterFramework` class
- `scripts/train.py` - Unified training entry point using `ModelFactory` pattern
- `scripts/experiment_manager.py` - Configuration management with `ExperimentConfig` dataclass
- `scripts/models/` - Model wrappers providing unified interfaces across architectures

**Key Design Patterns:**

- **Factory Pattern**: `ModelFactory.create_model()` abstracts model instantiation across 9+ architectures
- **Wrapper Pattern**: Each model family has a wrapper (e.g., `YOLOWrapper`, `YOLONASWrapper`) with consistent `.train()`, `.validate()`, `.predict()` methods
- **Configuration-Driven**: YAML configs in `configs/` control experiments, with `base.yaml` as the foundation

## Critical Workflows

### Training Pipeline

```bash
# Standard training workflow
python scripts/train.py --data data/sample_weeds.yaml --models yolov8n.pt yolov11n.pt --config configs/base.yaml

# Master framework (full research study)
python master_framework.py --experiment comprehensive_study --datasets all --models all
```

### Model Architecture Support

- **YOLO Family**: Uses `ultralytics` package directly in `YOLOWrapper`
- **YOLO-NAS**: Requires `super-gradients` dependency, wrapped in conditional imports
- **Other Models**: Feature conditional imports with availability flags (e.g., `YOLO_NAS_AVAILABLE`)

### Dataset Configuration

- **Path Convention**: Use absolute paths in YAML configs to prevent Ultralytics auto-relocation
- **Structure**: Follow `data/{dataset_name}.yaml` pattern with `path`, `train`, `val`, `names` keys
- **Example**: `data/dummy.yaml` shows minimal required structure

## Project-Specific Conventions

### File Organization

- `scripts/` - All executable Python modules (training, evaluation, utilities)
- `configs/` - YAML configuration files, with `experiments/` subfolder for study configs
- `results/` - Generated outputs, organized by study type (e.g., `master_study/`, `comprehensive_comparison/`)
- `models/` - Optimized/exported model artifacts

### Configuration Patterns

- **Base Config**: `configs/base.yaml` contains standard hyperparameters (epochs=1 for testing, 50 for production)
- **Experiment Configs**: Use `ExperimentConfig` dataclass in `experiment_manager.py` for complex studies
- **Device Handling**: Default to 'cuda' with fallback logic in model wrappers

### Model Wrapper Interface

All model wrappers must implement:

```python
def train(self, data: str, epochs: int, batch_size: int, device: str, **kwargs) -> Dict[str, Any]
def validate(self, data: str, weights: str, device: str, **kwargs) -> Dict[str, Any]
def predict(self, source: str, weights: str, **kwargs) -> results
```

### Error Handling Patterns

- **Conditional Imports**: Graceful degradation when optional dependencies unavailable
- **Model Availability**: Check availability flags before using advanced architectures
- **Path Resolution**: Always use `Path` objects and absolute paths for cross-platform compatibility

## Integration Points

### Dataset Integration

- **Download Scripts**: `scripts/download_datasets.py` handles dataset acquisition
- **Format Conversion**: `scripts/dataset_converter.py` standardizes formats across sources
- **Dummy Data**: `scripts/generate_dummy_data.py` creates test datasets for development

### Evaluation Framework

- **Advanced Evaluator**: `scripts/advanced_evaluator.py` provides comprehensive metrics beyond mAP
- **Energy Logging**: `scripts/energy_logger.py` measures power consumption during inference
- **Robustness Testing**: `scripts/perturb_eval.py` evaluates model stability under perturbations

### Export and Optimization

- **Model Pruning**: `scripts/prune.py` supports structured/unstructured pruning
- **Quantization**: `scripts/quantize.py` handles INT8 quantization and ONNX export
- **Format Export**: Unified export through model wrapper `.export()` methods

## Development Guidelines

### Adding New Models

1. Create wrapper in `scripts/models/{model_name}_wrapper.py`
2. Add to model lists in `scripts/train.py` (e.g., `YOLO_MODELS`, `EFFICIENTDET_MODELS`)
3. Update `ModelFactory.create_model()` with new model handling
4. Add conditional import with availability flag

### Configuration Management

- Use `ExperimentConfig` dataclass for complex experiments
- Store experiment configs in `configs/experiments/` with descriptive names
- Follow naming convention: `{study_type}_{date}.yaml`

### Testing and Validation

- Run `test_project.py` for comprehensive system validation
- Use `configs/base.yaml` with `epochs: 1` for quick smoke tests
- Validate model availability with `test_models_availability.py`

### Results Organization

- Follow pattern: `results/{study_type}/{timestamp}/`
- Generate summary JSONs: `training_summary.json`, `eval_summary.json`
- Use `scripts/advanced_visualizer.py` for publication-ready figures
