# Installation Guide

This guide covers different installation methods for the Vision Flow Framework.

## Quick Start (Recommended)

### Option 1: Minimal Installation (Core Features Only)

```bash
# Install core dependencies only
pip install -r requirements-minimal.txt
```

This installs:
- PyTorch and torchvision
- Ultralytics YOLO
- OpenCV, Pillow, albumentations
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn
- Essential utilities

✅ **Use this if**: You want to quickly test the framework or only need YOLO models.

### Option 2: Full Installation (With Optional Packages)

```bash
# Install core dependencies first
pip install -r requirements.txt

# Then install optional packages separately (to avoid conflicts)
pip install super-gradients timm transformers  # Advanced models
pip install wandb statsmodels pingouin  # Experiment tracking & statistics
pip install h5py  # HDF5 support
```

✅ **Use this if**: You want all features including YOLO-NAS, transformers, and experiment tracking.

## Installation Methods

### Method 1: Using pip (Standard)

```bash
# Clone the repository
git clone https://github.com/ahmeddi/Vision-Flow-Framework.git
cd Vision-Flow-Framework

# Install dependencies
pip install -r requirements-minimal.txt  # or requirements.txt
```

### Method 2: Using conda (Recommended for GPU)

```bash
# Create conda environment
conda create -n vff python=3.9
conda activate vff

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements-minimal.txt
```

### Method 3: Using venv

```bash
# Create virtual environment
python3 -m venv vff-env
source vff-env/bin/activate  # On Windows: vff-env\Scripts\activate

# Install dependencies
pip install -r requirements-minimal.txt
```

## Verify Installation

After installation, verify everything works:

```bash
# Test core imports
python3 -c "import torch; import ultralytics; import cv2; print('✅ Installation successful!')"

# Run project tests
python3 tests/test_models_availability.py
python3 tests/test_project.py
```

## Troubleshooting

### Issue: Dependency Conflicts

**Symptom**: `pip` takes forever resolving dependencies or shows conflicts.

**Solution**: Use minimal installation first, then add optional packages:

```bash
pip install -r requirements-minimal.txt
pip install super-gradients --no-deps  # Install without dependencies
```

### Issue: CUDA/GPU Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solution**: Reinstall PyTorch with CUDA support:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: OpenCV Import Error

**Symptom**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution** (Linux):
```bash
sudo apt-get install libgl1-mesa-glx
```

**Solution** (macOS): Already included, no action needed.

### Issue: numpy Version Conflict

**Symptom**: Errors about numpy 2.x incompatibility

**Solution**: Force install numpy 1.x:

```bash
pip install "numpy<2.0.0" --force-reinstall
```

## Package Versions

### Known Working Configuration

- Python: 3.9+
- PyTorch: 2.0.0 - 2.2.x
- Ultralytics: 8.0.0 - 8.3.x
- NumPy: 1.24.0 - 1.26.x (avoid 2.x)
- OpenCV: 4.8.0+

### Optional Package Notes

- **super-gradients**: Required for YOLO-NAS, may have dependency conflicts
- **transformers**: Required for DETR models, large package
- **wandb**: Optional experiment tracking, requires account
- **timm**: Required for EfficientDet backbones

## System Requirements

### Minimum

- OS: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- RAM: 8 GB
- Storage: 5 GB free space
- Python: 3.9+

### Recommended

- OS: Ubuntu 20.04+ or macOS 12+
- RAM: 16 GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (for training)
- Storage: 20 GB+ free space
- Python: 3.10+

## GPU Setup

### NVIDIA CUDA

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi
```

### Apple Silicon (M1/M2/M3)

PyTorch MPS backend is automatically used on Apple Silicon Macs.

```bash
# Check MPS availability
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Next Steps

After successful installation:

1. **Generate test data**: `python3 scripts/generate_dummy_data.py`
2. **Download models**: `python3 scripts/download_models.py --set essential`
3. **Run tests**: `python3 tests/test_project.py`
4. **Start training**: See [README.md](README.md) for training examples

## Getting Help

- Check [README.md](README.md) for usage examples
- Review [docs/guides/](docs/guides/) for detailed guides
- Open an issue on GitHub for bugs
- See [DEVELOPER_GUIDE_DETAILED.md](docs/guides/DEVELOPER_GUIDE_DETAILED.md) for architecture details

---

Last Updated: October 30, 2025
