# Comprehensive Weed Detection Benchmark

Complete research framework for comparing YOLOv8, YOLOv11, and state-of-the-art object detection models for real-time weed detection in precision agriculture.

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended for new users)

```bash
# Clone the repository
git clone https://github.com/ahmeddi/Vision-Flow-Framework.git
cd Vision-Flow-Framework

# Install dependencies
pip install -r requirements.txt

# Run automated setup (downloads models + datasets)
python setup_vff.py
```

This will automatically:

- ✅ Download essential pre-trained models (YOLOv8n, YOLO11n)
- ✅ Download sample datasets for testing
- ✅ Verify everything is working correctly

### Option 2: Manual Setup

#### 1. Download Pre-trained Models

```bash
# Download essential models (recommended for beginners)
python scripts/download_models.py --set essential

# Download research models (YOLOv8 + YOLO11 variants)
python scripts/download_models.py --set research

# Download specific models
python scripts/download_models.py --models yolov8n.pt yolov8s.pt yolo11n.pt

# See all available models
python scripts/download_models.py --list
```

**Available Model Sets:**

- `essential`: Basic models for quick testing (YOLOv8n, YOLO11n)
- `research`: Recommended for research (includes small/nano variants)
- `full_yolo`: All YOLO variants (v8 + v11)
- `all`: Every supported model architecture

#### 2. Download Datasets

```bash
# Generate demo dataset (for testing)
python scripts/generate_dummy_data.py --n_train 20 --n_val 10

# Download real datasets
python scripts/download_datasets.py --datasets deepweeds sample_weeds --sample 50
```

#### 3. Verify Setup

```bash
# Test model availability
python scripts/test_models_availability.py

# Test datasets
python scripts/test_datasets_availability.py
```

### 2. Train Models

```bash
# Single model training
python scripts/train.py --data data/dummy.yaml --models yolov8n.pt --config configs/base.yaml

# Multiple models benchmark
python scripts/train.py --data data/sample_weeds.yaml --models yolov8n.pt yolov8s.pt yolov11n.pt --config configs/base.yaml
```

### 3. Evaluate Performance

```bash
# Basic evaluation (mAP, FPS, latency)
python scripts/evaluate.py --models_dir results/runs --device cuda

# Energy consumption analysis
python scripts/energy_logger.py --models results/runs/yolov8n/weights/best.pt --device cuda --n_images 100

# Robustness testing (perturbations)
python scripts/perturb_eval.py --model results/runs/yolov8n/weights/best.pt --data data/sample_weeds.yaml
```

### 4. Model Optimization

```bash
# Pruning (structured/unstructured)
python scripts/prune.py --model results/runs/yolov8n/weights/best.pt --output models/yolov8n_pruned.pt --method unstructured --amount 0.3

# Quantization (INT8, ONNX export)
python scripts/quantize.py --model results/runs/yolov8n/weights/best.pt --formats onnx tensorrt --benchmark
```

## 📊 Project Structure

```
vff/
├── configs/
│   └── base.yaml              # Training hyperparameters
├── data/
│   ├── dummy/                 # Generated test dataset
│   ├── sample_weeds/          # Downloaded sample dataset
│   ├── dummy.yaml             # YOLO dataset config
│   └── sample_weeds.yaml      # Real dataset config
├── scripts/
│   ├── download_datasets.py   # Dataset acquisition
│   ├── generate_dummy_data.py # Test data generation
│   ├── train.py               # Unified training script
│   ├── evaluate.py            # Performance evaluation
│   ├── energy_logger.py       # Energy consumption measurement
│   ├── perturb_eval.py        # Robustness testing
│   ├── prune.py               # Model pruning
│   └── quantize.py            # Model quantization
├── results/
│   ├── runs/                  # Training outputs
│   ├── tables/               # Results tables
│   ├── figures/              # Generated plots
│   ├── training_summary.json # Training metrics
│   └── eval_summary.json     # Evaluation results
├── models/                    # Optimized models
├── paper_outline.md          # Research paper structure
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎯 Supported Models

### YOLO Family

- **YOLOv8**: n, s, m, l, x variants
- **YOLOv11**: n, s, m, l, x variants
- **YOLOv7**: Various configurations
- **YOLO-NAS**: S, M, L variants

### Other Architectures

- **EfficientDet**: D0-D7 (planned)
- **DETR/RT-DETR**: Transformer-based (planned)
- **PP-YOLOE**: PaddlePaddle implementation (planned)

## 📁 Datasets

### Supported Datasets

1. **DeepWeeds** - 17,509 images, 8 weed species from northern Australia
2. **AgML Weed25** - 25 weed species (planned)
3. **CWFID** - Crop/Weed Field Image Dataset (planned)
4. **Sample Demo** - COCO8 subset for testing

### Dataset Format

All datasets are converted to YOLO format:

- `images/train/`, `images/val/`, `images/test/`
- `labels/train/`, `labels/val/`, `labels/test/`
- Class definitions in YAML files

## 🔧 Key Features

### 1. Unified Training Framework

- Consistent hyperparameters across models
- Reproducible training with fixed seeds
- Automatic mixed precision (AMP)
- Advanced augmentations (Mosaic, MixUp, HSV)

### 2. Comprehensive Evaluation

- **Accuracy**: mAP@0.5, mAP@0.5:0.95
- **Speed**: FPS, latency (mean/p95)
- **Efficiency**: Model size, memory usage
- **Energy**: Power consumption (J/frame)
- **Robustness**: Performance under perturbations

### 3. Model Optimization

- **Pruning**: Structured/unstructured parameter removal
- **Quantization**: INT8 conversion for edge deployment
- **Export**: ONNX, TensorRT formats

### 4. Robustness Testing

Environmental perturbations:

- Brightness/contrast variations
- Gaussian noise
- Motion blur
- Gamma correction

## 📈 Metrics & Analysis

### Primary Metrics

- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: mAP averaged over IoU 0.5-0.95
- **FPS**: Frames per second (inference speed)
- **Latency**: Inference time per image
- **Energy**: Joules per frame
- **Size**: Model file size (MB)

### Statistical Analysis

- Bootstrap confidence intervals
- Wilcoxon signed-rank tests
- Spearman correlation analysis
- Effect size reporting

## 🏃‍♂️ Usage Examples

### Training Multiple Models

```python
# Train YOLO variants on custom dataset
python scripts/train.py \\
    --data data/my_weeds.yaml \\
    --models yolov8n.pt yolov8s.pt yolov11n.pt \\
    --config configs/base.yaml \\
    --output results/benchmark
```

### Energy Benchmarking

```python
# Compare energy consumption
python scripts/energy_logger.py \\
    --models results/runs/*/weights/best.pt \\
    --device cuda \\
    --n_images 200 \\
    --output results/energy_comparison.json
```

### Model Compression

```python
# Prune and quantize for edge deployment
python scripts/prune.py --model model.pt --output pruned.pt --amount 0.4
python scripts/quantize.py --model pruned.pt --formats onnx tflite
```

## 🔬 Research Applications

### Paper Contributions

1. **Multi-generational YOLO comparison** (v8 vs v11)
2. **Energy-aware benchmarking** for agricultural robotics
3. **Robustness evaluation** under field conditions
4. **Optimization techniques** for edge deployment

### Reproducibility

- Fixed random seeds (42)
- Deterministic operations where possible
- Version pinning (`requirements.txt`)
- Detailed hyperparameter logging

## 🚧 Current Status

### ✅ Completed

- [x] Project structure and framework
- [x] Dummy dataset generation
- [x] Basic training pipeline (YOLOv8)
- [x] Evaluation framework
- [x] Energy logging infrastructure
- [x] Pruning implementation
- [x] Quantization framework
- [x] Robustness testing structure

### 🔄 In Progress

- [ ] Real dataset integration (DeepWeeds URLs)
- [ ] Multi-model training validation
- [ ] Statistical analysis implementation

### 📋 Planned

- [ ] Additional model families (DETR, EfficientDet)
- [ ] Cross-dataset evaluation
- [ ] Automated report generation
- [ ] Docker containerization

## 🤝 Contributing

### Adding New Models

1. Create model wrapper in `scripts/train.py`
2. Add evaluation support in `scripts/evaluate.py`
3. Update configuration files

### Adding New Datasets

1. Add dataset info to `scripts/download_datasets.py`
2. Implement format conversion if needed
3. Create YAML configuration file

## 📚 Citations

```bibtex
@article{your_paper_2024,
  title={Comparative Evaluation of YOLOv8, YOLOv11, and State-of-the-Art Object Detection Models for Real-Time Weed Detection in Precision Agriculture},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License. See individual dataset licenses for data usage terms.

## 🔗 Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DeepWeeds Dataset Paper](https://www.nature.com/articles/s41598-018-38343-3)
- [AgML Agricultural Dataset Collection](https://github.com/AgML/AgML)

---

**Last Updated**: August 2025  
**Version**: 0.1.0  
**Framework**: PyTorch + Ultralytics YOLO
