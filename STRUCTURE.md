# Vision Flow Framework - Directory Structure

```
Vision-Flow-Framework/
│
├── 📄 master_framework.py              # Main orchestration script
├── 📄 requirements.txt                 # Python dependencies  
├── 📄 README.md                        # Main documentation (English)
├── 📄 README_FR.md                     # Main documentation (French)
├── 📄 PROJECT_CLEANUP_SUMMARY.md       # This cleanup summary
│
├── 📁 configs/                         # Configuration files
│   ├── base.yaml                       # Base training configuration
│   └── experiments/                    # Experiment-specific configs
│
├── 📁 data/                            # Datasets
│   ├── *.yaml                          # Dataset configuration files
│   ├── dummy/                          # Test dataset
│   ├── sample_weeds/                   # Sample dataset
│   ├── deepweeds/                      # DeepWeeds dataset
│   ├── cwd30/                          # CWD30 dataset
│   ├── weed25/                         # Weed25 dataset
│   └── weedsgalore/                    # WeedsGalore dataset
│
├── 📁 scripts/                         # Executable scripts
│   ├── train.py                        # Main training script
│   ├── evaluate.py                     # Model evaluation
│   ├── download_models.py              # Model downloading
│   ├── download_datasets.py            # Dataset downloading
│   ├── generate_dummy_data.py          # Test data generation
│   ├── setup_vff.py                    # VFF setup utility
│   ├── setup_colab.py                  # Colab setup utility
│   ├── energy_logger.py                # Energy measurement
│   ├── perturb_eval.py                 # Robustness testing
│   ├── prune.py                        # Model pruning
│   ├── quantize.py                     # Model quantization
│   ├── advanced_evaluator.py           # Advanced metrics
│   ├── advanced_visualizer.py          # Visualization tools
│   ├── statistical_analysis.py         # Statistical analysis
│   ├── comprehensive_benchmark.py      # Full benchmarking
│   ├── methodology_benchmark.py        # Methodology testing
│   ├── real_data_analysis.py           # Real data analysis
│   ├── real_data_comparison.py         # Real data comparison
│   ├── generate_discussion_plots.py    # Plot generation
│   ├── generate_performance_tables.py  # Table generation
│   ├── experiment_manager.py           # Experiment management
│   ├── dataset_converter.py            # Dataset format conversion
│   └── models/                         # Model wrapper classes
│       ├── yolo_wrapper.py
│       ├── yolonas_wrapper.py
│       └── ...
│
├── 📁 tests/                           # Test suite
│   ├── README.md                       # Test documentation
│   ├── __init__.py                     # Package initialization
│   ├── test_project.py                 # Comprehensive system test
│   ├── test_models_availability.py     # Model availability test
│   ├── test_datasets_availability.py   # Dataset availability test
│   ├── validate_paper_setup.py         # Paper setup validation
│   ├── test_yolo11.py                  # YOLOv11 specific tests
│   ├── test_bytes.py                   # Byte handling tests
│   ├── test_direct.py                  # Direct execution tests
│   └── test_results.json               # Test results output
│
├── 📁 docs/                            # Documentation
│   ├── README.md                       # Documentation index
│   │
│   ├── 📁 guides/                      # User guides
│   │   ├── VFF_SIMPLE_GUIDE.md         # Quick start guide
│   │   ├── VFF_GUIDE_SIMPLE_FR.md      # Guide rapide (français)
│   │   ├── DEVELOPER_GUIDE_DETAILED.md # Developer guide
│   │   ├── INSTALLATION_ARCHITECTURES_COMPLETE.md
│   │   ├── MODEL_DOWNLOAD_GUIDE.md
│   │   ├── GUIDE_ARCHITECTURES_DISPONIBLES.md
│   │   ├── VFF_GPU_NOTEBOOK_GUIDE.md
│   │   └── COLAB_GUIDE.md
│   │
│   ├── 📁 notebooks/                   # Jupyter notebooks
│   │   ├── VFF_Colab_Setup.ipynb       # Colab setup
│   │   └── VFF_GPU_Training_Complete.ipynb  # GPU training
│   │
│   ├── 📁 paper/                       # Research paper materials
│   │   ├── paper_outline.md            # Paper structure
│   │   ├── ARTICLE_INTRODUCTION.md     # Introduction section
│   │   ├── PROJECT_PAPER_ANALYSIS.md   # Analysis for publication
│   │   ├── RESULTATS_ET_DISCUSSION.md  # Results (French)
│   │   ├── SECTION_RESULTATS_DISCUSSION_COMPLETE.md
│   │   └── SECTION_RESULTATS_DISCUSSION_COMPLETE.html
│   │
│   └── 📁 reports/                     # Analysis reports
│       ├── METHODOLOGY_COMPLETE_GUIDE.md
│       ├── METHODOLOGY_VALIDATION_REPORT.md
│       ├── TRAINING_RESULTS_SUMMARY.md
│       ├── PROJECT_COMPLETION_SUMMARY.md
│       ├── VISUALIZATION_REPORT.md
│       ├── IMPROVEMENT_ROADMAP.md
│       ├── COLAB_FIX_SUMMARY.md
│       └── real_data_comparison.log
│
└── 📁 results/                         # Generated outputs
    ├── runs/                           # Training outputs
    ├── tables/                         # Results tables
    ├── figures/                        # Generated plots
    └── *.json                          # Summary files
```

## 📊 File Count Summary

- **Root Directory**: 5 files (clean!)
- **Tests**: 8 test files + documentation
- **Documentation**: 20+ documents organized by category
- **Scripts**: 30+ executable scripts
- **Datasets**: 6 dataset directories configured

## 🎯 Organization Principles

1. **Root**: Only essential files (README, requirements, main script)
2. **tests/**: All testing and validation scripts
3. **docs/**: All documentation, organized by type
4. **scripts/**: All executable Python scripts
5. **configs/**: Configuration files only
6. **data/**: Datasets only
7. **results/**: Generated outputs only

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_project.py

# Train models
python scripts/train.py --data data/dummy.yaml --models yolov8n.pt

# View documentation
cat docs/README.md
```

---

**Last Updated**: October 30, 2025
**Version**: 2.0.0 (Post-Cleanup)
