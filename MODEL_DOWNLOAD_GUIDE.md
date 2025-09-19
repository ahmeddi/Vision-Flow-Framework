# Model Download Guide for VFF (Vision Flow Framework)

## 🎯 Quick Start for New Users

When you clone the VFF project, you'll need to download pre-trained models to start training. Here are your options:

### Option 1: Automated Setup (Recommended)

```bash
# One command to set up everything
python setup_vff.py
```

This will:

- ✅ Download essential models (YOLOv8n, YOLO11n)
- ✅ Download sample datasets
- ✅ Verify everything works

### Option 2: Manual Model Download

```bash
# Download essential models (fastest)
python scripts/download_models.py --set essential

# Download research models (recommended)
python scripts/download_models.py --set research

# Download all YOLO models
python scripts/download_models.py --set full_yolo

# Download everything
python scripts/download_models.py --all
```

### Option 3: Specific Models

```bash
# Download specific models
python scripts/download_models.py --models yolov8n.pt yolo11n.pt yolov8s.pt

# See all available models
python scripts/download_models.py --list
```

## 📋 Available Model Sets

| Set Name    | Models               | Size     | Description              |
| ----------- | -------------------- | -------- | ------------------------ |
| `essential` | YOLOv8n, YOLO11n     | ~11 MB   | Quick testing            |
| `research`  | YOLOv8n/s, YOLO11n/s | ~47 MB   | Research work            |
| `full_yolo` | All YOLO variants    | ~400 MB  | Complete YOLO collection |
| `all`       | Every model          | ~400+ MB | Everything available     |

## 🔍 Verify Setup

```bash
# Test if models are working
python test_models_availability.py

# Test specific training
python scripts/train.py --models yolov8n.pt --data data/sample_weeds.yaml --epochs 1
```

## 🛠️ Troubleshooting

### Models Not Found

```bash
# Make sure you're in the project root
ls -la *.pt

# If no .pt files, download them:
python scripts/download_models.py --set essential
```

### Import Errors

```bash
# Install dependencies
pip install -r requirements.txt

# Specifically for model handling
pip install torch ultralytics requests tqdm
```

### Download Failures

```bash
# Try force re-download
python scripts/download_models.py --set essential --force

# Check internet connection and try again
```

## 📁 Where Models Are Stored

Models are downloaded to the **project root directory** (same folder as `scripts/`):

```
Vision-Flow-Framework/
├── yolov8n.pt          ← Downloaded models here
├── yolo11n.pt          ← Downloaded models here
├── scripts/            ← Training scripts
├── data/              ← Datasets
└── results/           ← Training results
```

## 🚀 What to Do After Download

1. **Verify setup**: `python test_models_availability.py`
2. **Quick test**: `python scripts/train.py --models yolov8n.pt --data data/sample_weeds.yaml --epochs 1`
3. **Full training**: `python scripts/run_comprehensive_training.py`

## 💡 Pro Tips

- Start with `essential` models for quick testing
- Use `research` models for actual research work
- The `yolov8n.pt` and `yolo11n.pt` models are fastest for development
- Larger models (s, m, l, x) give better accuracy but take longer to train

## 🆘 Need Help?

1. Check `README.md` for full documentation
2. Run `python scripts/download_models.py --help`
3. Look at example commands in the scripts
4. Check the VFF documentation files

## 🎉 Success!

Once you see `✅ Models downloaded successfully`, you're ready to start training world-class weed detection models!
