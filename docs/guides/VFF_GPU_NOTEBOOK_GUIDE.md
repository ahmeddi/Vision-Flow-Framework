# Vision Flow Framework - GPU Training Notebook Guide

## 📓 Notebook: `VFF_GPU_Training_Complete.ipynb`

This comprehensive Jupyter notebook provides a complete end-to-end solution for training and comparing multiple object detection models for weed detection on GPU systems.

## 🎯 What's Included

### Complete Model Coverage
- **YOLOv8**: All 5 variants (n, s, m, l, x)
- **YOLOv11**: All 5 variants (n, s, m, l, x)
- **YOLO-NAS**: Small, Medium, Large
- **YOLOX**: Tiny, S, M, L, X
- **YOLOv7**: Standard variant
- **PP-YOLOE**: S, M, L, X variants
- **EfficientDet**: D0-D7 variants
- **RT-DETR**: L, X variants
- **DETR**: Original transformer model

### Dataset Support
- ✅ Weed25
- ✅ DeepWeeds
- ✅ CWD30
- ✅ WeedsGalore (multispectral UAV)
- ✅ Sample Weeds (for testing)

### Comprehensive Analysis
1. **Performance Metrics**
   - mAP@50 and mAP@50-95
   - Precision, Recall, F1-Score
   - Confusion matrices

2. **Speed Benchmarking**
   - FPS (Frames Per Second)
   - Inference time (ms)
   - Batch processing performance

3. **Resource Analysis**
   - Model size (MB)
   - Parameter count
   - GPU memory consumption
   - Energy estimation

4. **Robustness Testing**
   - Brightness variations
   - Darkness conditions
   - Contrast changes
   - Blur effects
   - Noise resistance
   - Rotation invariance

5. **Visualization**
   - Side-by-side detection comparisons
   - Performance charts
   - Memory usage graphs
   - Speed comparisons

6. **Model Export**
   - ONNX format
   - TorchScript
   - TensorRT (NVIDIA GPUs)
   - CoreML (iOS/macOS)
   - TFLite (mobile/embedded)

## 🚀 Quick Start

### 1. Upload to Remote GPU Server

```bash
# Upload notebook to your GPU server
scp VFF_GPU_Training_Complete.ipynb user@gpu-server:/path/to/workspace/

# Or use Jupyter Lab/Hub interface to upload
```

### 2. Configure Training Parameters

Open the notebook and modify Section 6:

```python
TRAINING_CONFIG = {
    'epochs': 50,              # ← Change this (1 for testing, 50-100 for production)
    'batch_size': 16,          # ← Adjust based on GPU memory
    'img_size': 640,           # ← Image size (416, 512, 640, 800, 1024)
    'dataset': 'sample_weeds', # ← Select dataset
    # ... more parameters
}
```

### 3. Run All Cells

You can run the entire notebook or execute specific sections:

- **Section 1-3**: Environment setup and GPU verification
- **Section 4-6**: Dataset preparation and configuration
- **Section 7-11**: Model training
- **Section 12-15**: Evaluation and analysis
- **Section 16-18**: Robustness, visualization, and export

## 📊 Customization Options

### Training Presets

```python
# Quick test (1 epoch, small batch)
TRAINING_CONFIG.update(QUICK_TEST_CONFIG)

# Standard training (50 epochs)
TRAINING_CONFIG.update(STANDARD_CONFIG)

# Full training (100 epochs, larger batch)
TRAINING_CONFIG.update(FULL_TRAINING_CONFIG)
```

### Select Specific Models

```python
# Train only specific YOLOv8 variants
YOLOV8_MODELS = ['yolov8n.pt', 'yolov8s.pt']  # Only nano and small

# Train only specific YOLOv11 variants
YOLOV11_MODELS = ['yolov11m.pt', 'yolov11l.pt']  # Only medium and large
```

### Choose Dataset

```python
TRAINING_CONFIG['dataset'] = 'weed25'      # Weed25 dataset
TRAINING_CONFIG['dataset'] = 'deepweeds'   # DeepWeeds dataset
TRAINING_CONFIG['dataset'] = 'cwd30'       # CWD30 dataset
TRAINING_CONFIG['dataset'] = 'weedsgalore' # WeedsGalore dataset
```

### Adjust for GPU Memory

If you encounter GPU memory errors:

```python
TRAINING_CONFIG['batch_size'] = 8    # Reduce from 16
TRAINING_CONFIG['img_size'] = 416    # Reduce from 640
TRAINING_CONFIG['workers'] = 4       # Reduce from 8
```

## 📁 Output Structure

After running the notebook, results are organized as:

```
results/
├── yolov8/
│   ├── yolov8n/
│   ├── yolov8s/
│   └── ...
├── yolov11/
│   ├── yolov11n/
│   └── ...
├── model_comparison.csv          # Metrics comparison table
├── model_comparison.png          # Performance charts
├── fps_comparison.png            # Speed comparison
├── memory_analysis.csv           # Memory usage data
├── memory_comparison.png         # Memory charts
├── robustness_analysis.png       # Robustness results
├── detection_comparison_*.png    # Visual comparisons
└── comprehensive_report.json     # Complete summary

models/
└── exported/                     # Exported models (ONNX, etc.)
```

## 💡 Best Practices

### 1. Start with Quick Test
```python
TRAINING_CONFIG['epochs'] = 1
TRAINING_CONFIG['batch_size'] = 8
TRAINING_CONFIG['dataset'] = 'sample_weeds'
```
Run once to verify everything works.

### 2. Scale Up Gradually
```python
TRAINING_CONFIG['epochs'] = 10
TRAINING_CONFIG['batch_size'] = 16
```
Test with small epochs first.

### 3. Full Training
```python
TRAINING_CONFIG['epochs'] = 100
TRAINING_CONFIG['batch_size'] = 32  # If GPU allows
```
Use for final production models.

### 4. Monitor GPU Usage
Check GPU utilization during training:
```python
# In a separate cell or terminal
!nvidia-smi
```

### 5. Save Checkpoints
The notebook automatically saves model checkpoints during training.

## 🔧 Troubleshooting

### Problem: CUDA Out of Memory
**Solution:**
- Reduce batch_size to 8 or 4
- Reduce img_size to 416
- Train only nano (n) or small (s) models

### Problem: Training Too Slow
**Solution:**
- Ensure GPU is being used (check device='cuda')
- Reduce number of workers if CPU is bottleneck
- Use smaller models or datasets for testing

### Problem: Model Import Errors
**Solution:**
- Some models require additional packages
- Install missing dependencies as shown in error messages
- Some models (YOLO-NAS, DETR) need custom setup

### Problem: Dataset Not Found
**Solution:**
- Ensure dataset YAML files exist in `data/` directory
- Check paths in YAML files are absolute
- Use `sample_weeds` dataset for testing

## 📈 Performance Expectations

### Training Time (per epoch, sample_weeds)
- **Nano models (n)**: ~30-60 seconds
- **Small models (s)**: ~1-2 minutes
- **Medium models (m)**: ~3-5 minutes
- **Large models (l)**: ~5-10 minutes
- **Extra-large models (x)**: ~10-20 minutes

*Times vary based on GPU, dataset size, and image size*

### GPU Memory Usage
- **Nano models**: 2-4 GB
- **Small models**: 4-6 GB
- **Medium models**: 6-10 GB
- **Large models**: 10-16 GB
- **Extra-large models**: 16-24 GB

## 🎓 Learning Path

### Beginner
1. Run Sections 1-3 (setup and verification)
2. Run Section 6 with `QUICK_TEST_CONFIG`
3. Train one YOLOv8n model (Section 7)
4. Evaluate and visualize (Sections 12-13)

### Intermediate
1. Train all YOLOv8 variants
2. Train all YOLOv11 variants
3. Compare performance metrics
4. Test inference speed
5. Export best models

### Advanced
1. Train all available models
2. Comprehensive robustness testing
3. Custom perturbation analysis
4. Multi-dataset evaluation
5. Production deployment preparation

## 🔗 Integration with VFF

This notebook integrates seamlessly with the Vision Flow Framework:

- Uses same dataset structure (`data/*.yaml`)
- Compatible with VFF model wrappers
- Results format matches VFF outputs
- Can be used alongside `scripts/train.py`

## 📝 Citation

If you use this notebook in your research, please cite:

```
Vision Flow Framework - Comprehensive Weed Detection Study
Comparative Analysis of YOLOv8, YOLOv11, and State-of-the-Art Object Detectors
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review VFF documentation
3. Check GitHub issues
4. Refer to Ultralytics documentation for YOLO-specific issues

---

**Version**: 1.0  
**Created**: October 2025  
**Compatible with**: CUDA 11.8+, PyTorch 2.0+, Ultralytics 8.0+
