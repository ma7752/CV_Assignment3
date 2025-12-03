# File Organization Guide

## ✅ KEEP - Project Structure Files

### Root Directory
```
├── .gitignore              # Git ignore rules
├── pytest.ini              # Pytest configuration
├── README.md              # Project documentation
├── requirements.txt        # Python dependencies
├── data/                  # Data directory (gitignored)
├── outputs/               # Output directory (gitignored)
├── venv/                  # Virtual environment (gitignored)
```

### Source Code (`src/`)
```
src/
├── __init__.py            # Package marker
├── config.py              # Configuration settings
├── model.py               # MovedObjectDETR model
├── dataloader.py          # Dataset class
├── train.py               # Training script
├── evaluate.py            # Evaluation metrics
├── utils.py               # Utility functions
```

### Tests (`tests/`)
```
tests/
├── __init__.py            # Package marker
├── conftest.py            # Pytest fixtures
├── README.md             # Test documentation
├── test_model.py          # Model unit tests
├── test_config.py         # Config unit tests
├── test_dataloader.py     # Dataloader unit tests
```

### Scripts (`scripts/`)
```
scripts/
├── run_training.sh        # Training script
├── run_evaluation.sh      # Evaluation script
```

## 🗑️ DELETE - Unnecessary Files

### Debug/Temporary Files
```bash
# Delete these debug scripts
rm check_detr_structure.py
rm debug_detr.py
rm src/inspect_model.py
rm src/test_dataloader.py    # Replaced by tests/test_dataloader.py
```

### Cache Directories
```bash
# These are auto-generated, safe to delete
rm -rf __pycache__/
rm -rf src/__pycache__/
rm -rf .pytest_cache/
rm -rf .qodo/
```

## 📁 MOVE - Reorganize Manual Test Scripts

### Keep in Root (Manual Integration Tests)
```
test_model_detailed.py     # Manual model test with detailed output
test_real_data.py          # Manual test with real VIRAT images
```

These are NOT pytest tests - they're manual integration test scripts.
Run them directly:
```bash
python test_model_detailed.py
python test_real_data.py
```

## 🔧 Commands to Clean Up

### Step 1: Delete debug files
```bash
cd /mnt/c/Users/muham/OneDrive/Desktop/ComputerVision/CV_Assignment3

# Delete debug scripts
rm check_detr_structure.py
rm debug_detr.py
rm src/inspect_model.py
rm src/test_dataloader.py
```

### Step 2: Clean cache
```bash
# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Remove pytest cache
rm -rf .pytest_cache/
rm -rf .qodo/
```

### Step 3: Verify structure
```bash
# List what's left
ls -la
ls -la src/
ls -la tests/
```

## 📝 Final Structure

After cleanup, your project should look like:

```
CV_Assignment3/
├── .git/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── test_model_detailed.py        # Manual test
├── test_real_data.py             # Manual test
├── data/                         # Gitignored
├── outputs/                      # Gitignored
├── venv/                         # Gitignored
├── scripts/
│   ├── run_training.sh
│   └── run_evaluation.sh
├── src/
│   ├── config.py
│   ├── dataloader.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── README.md
    ├── test_config.py
    ├── test_dataloader.py
    └── test_model.py
```

## 🎯 Usage Summary

### Run Automated Tests (Pytest)
```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest tests/test_model.py  # Specific file
```

### Run Manual Tests
```bash
python test_model_detailed.py  # Quick model verification
python test_real_data.py        # Test with actual VIRAT data
```

### Development
```bash
python src/train.py       # Train model
python src/evaluate.py    # Evaluate model
```
