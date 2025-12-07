# Quick Reference Guide

## Running Scripts

All scripts now work from the project root directory with proper path handling.

### Training Models

```bash
# Baseline CNN
python scripts/training/train_country_model.py

# Multi-Task CNN with Attention
python scripts/training/train_multitask.py

# Hybrid CNN-CLIP (requires CLIP)
python scripts/training/train_hybrid_cnn_vit.py

# CLIP-Only (requires CLIP)
python scripts/training/train_clip_only.py
```

### Evaluation & Comparison

```bash
# Compare baseline vs multi-task
python scripts/evaluation/compare_models.py

# Compare all 4 models
python scripts/evaluation/compare_all_models.py

# Test random samples
python scripts/evaluation/test_random_images.py
```

### Visualization

```bash
# Generate attention heatmaps
python scripts/visualization/visualize_attention.py
```

## How It Works

Each script now:
1. Automatically detects the project root directory
2. Adds the root to Python's path for imports
3. Uses absolute paths relative to project root
4. Works from any directory when run from project root

## Path Structure

- `DATA_DIR`: `data/gsv_50k/compressed_dataset/`
- `MODEL_DIR`: `models/`
- `OUTPUT_DIR`: `outputs/metrics/` or `outputs/attention_maps/`

All paths are automatically resolved relative to the project root, so you can run scripts from anywhere within the project.

## Imports

Scripts can import from each other:
```python
from scripts.training.train_multitask import MultiTaskGeoModel
from scripts.training.train_hybrid_cnn_vit import HybridCNNViT
```

## Notes

- CLIP-based models (Hybrid and CLIP-Only) require CLIP installation
- If CLIP is not installed, comparison scripts will work with approximate parameter counts
- All outputs are saved relative to project root
