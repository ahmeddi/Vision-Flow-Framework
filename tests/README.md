# Vision Flow Framework Test Suite

This directory contains all tests for the Vision Flow Framework (VFF).

## 🧪 Available Tests

### System Tests

- **test_project.py** - Comprehensive system validation
  - Tests core functionality
  - Validates model availability
  - Checks dataset integration
  - Verifies script execution

### Model Tests

- **test_models_availability.py** - Validates model files and architecture support
  - Checks for downloaded models
  - Verifies model loading
  - Tests each architecture wrapper

### Dataset Tests

- **test_datasets_availability.py** - Validates dataset configuration and availability
  - Checks dataset YAML files
  - Verifies dataset structure
  - Tests data loading

### Integration Tests

- **validate_paper_setup.py** - Validates research paper setup
  - Checks experiment configurations
  - Verifies result reproducibility
  - Tests complete pipeline

### Legacy Tests

- **test_bytes.py** - Byte handling tests
- **test_direct.py** - Direct execution tests
- **test_yolo11.py** - YOLOv11 specific tests

## 🚀 Running Tests

### Run All Tests

```bash
# Run comprehensive system test
python tests/test_project.py
```

### Run Specific Tests

```bash
# Test model availability
python tests/test_models_availability.py

# Test datasets
python tests/test_datasets_availability.py

# Validate paper setup
python tests/validate_paper_setup.py
```

### Run with pytest (if installed)

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_project.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

## 📊 Test Results

Test results are stored in:

- **test_results.json** - JSON output of test runs
- Console output for immediate feedback

## 🔧 Adding New Tests

When adding new tests:

1. Create test file with `test_` prefix
2. Follow existing test patterns
3. Document what the test validates
4. Add to this README

## ✅ Test Coverage

Current test coverage includes:

- ✅ Model loading and initialization
- ✅ Dataset configuration validation
- ✅ Training pipeline execution
- ✅ Evaluation metrics computation
- ✅ Export and optimization workflows

### Planned Test Coverage

- 🔄 Unit tests for individual modules
- 🔄 Integration tests for full workflows
- 🔄 Performance benchmarking tests
- 🔄 Error handling and edge cases

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **Missing Models**: Run `python scripts/download_models.py --set essential`
3. **Missing Datasets**: Run `python scripts/generate_dummy_data.py`

### Debug Mode

Run tests with verbose output:

```bash
python tests/test_project.py --verbose
```

---

Last Updated: October 2025
