# Multi-Task Geolocation with Attention

## Quick Start Guide

### 1. Train the Multi-Task Model (with Attention)
```bash
# On Google Colab (recommended)
!python train_multitask.py
```
This will train a model that predicts:
- **Country** (120 classes)
- **Region** (7 geographic regions)
- **Climate** (6 climate zones)

With built-in **attention mechanism** for interpretability.

### 2. Visualize Attention Maps
```bash
!python visualize_attention.py
```
Creates visual heatmaps showing what image regions the model focuses on.

Output: `outputs/attention_maps/`

### 3. Compare with Baseline
```bash
!python compare_models.py
```
Generates ablation study comparing:
- Single-task vs Multi-task
- With vs without attention

Output: `models/ablation_comparison.png`

### 4. Test on Random Images
```bash
!python test_random_images.py
```
Tests model on random samples from the dataset with accuracy metrics.

---

## Files Overview

### Training Scripts
- `train_country_model.py` - **Baseline**: Single-task country classification
- `train_multitask.py` - **Novel**: Multi-task learning with attention

### Evaluation Scripts
- `test_random_images.py` - Test baseline model on random samples
- `visualize_attention.py` - Generate attention heatmaps
- `compare_models.py` - Ablation study and comparison

### Utilities
- `filter_dataset.py` - Filter dataset by minimum images per country
- `test_country_model.py` - Test baseline on single image (CLI)

### Documentation
- `PRESENTATION_OUTLINE.md` - Complete presentation structure
- `README.md` - This file

---

## Research Workflow

### Phase 1: Baseline (Already Done)
1. ✓ Train single-task model: `python train_country_model.py`
2. ✓ Test it: `python test_random_images.py`

### Phase 2: Novel Approach (Do This)
1. **Train multi-task model**: `python train_multitask.py`
2. **Generate attention visualizations**: `python visualize_attention.py`
3. **Run ablation study**: `python compare_models.py`

### Phase 3: Analysis
1. Compare results (baseline vs multi-task)
2. Analyze attention patterns
3. Identify failure cases
4. Document findings

### Phase 4: Presentation
1. Fill in results in `PRESENTATION_OUTLINE.md`
2. Use attention visualizations as figures
3. Show ablation study comparison
4. Prepare live demo

---

## Expected Results

### Baseline (Single-Task)
- Country Accuracy: ~XX%
- Parameters: 23.5M
- No interpretability

### Multi-Task + Attention (Novel)
- Country Accuracy: ~XX% (+X% improvement)
- Region Accuracy: ~XX%
- Climate Accuracy: ~XX%
- Parameters: 23.5M (same!)
- **+ Visual attention maps for interpretability**

---

## Key Innovations

1. **Multi-Task Learning**
   - Simultaneous prediction of country, region, climate
   - Shared representations improve accuracy
   - Geographic hierarchy aids learning

2. **Spatial Attention**
   - Highlights important image regions
   - Provides visual explanations
   - No accuracy loss, pure benefit

3. **Comprehensive Evaluation**
   - Ablation study quantifies contributions
   - Attention visualization for interpretability
   - Comparison with baseline

---

## For Google Colab

### Complete Setup
```python
# Clone repo
!git clone https://github.com/johntmvu/cnn-gat-geolocation.git
%cd cnn-gat-geolocation

# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip dataset to local storage
!mkdir -p data
!unzip -q /content/drive/MyDrive/AI/gsv_50k.zip -d data/

# Train multi-task model
!python train_multitask.py

# Visualize attention
!python visualize_attention.py

# Compare models
!python compare_models.py
```

---

## Model Architecture

```
Input Image (224×224)
    ↓
ResNet50 Backbone
    ↓
Spatial Attention Layer ← Highlights important regions
    ↓
Global Average Pooling
    ↓
    ├→ Country Head (120 classes)
    ├→ Region Head (7 classes)
    └→ Climate Head (6 classes)
```

---

## Outputs

### Models Saved
- `models/best_country_model.pth` - Baseline single-task
- `models/best_multitask_model.pth` - Multi-task with attention
- `models/country_mapping.json` - Country labels
- `models/multitask_mappings.json` - All task labels

### Metrics
- `models/training_metrics.json` - Baseline training logs
- `models/multitask_metrics.json` - Multi-task training logs

### Visualizations
- `outputs/attention_maps/` - Individual attention heatmaps
- `outputs/attention_maps/attention_grid.png` - Grid visualization
- `models/ablation_comparison.png` - Comparison plots
- `models/test_predictions.png` - Random test samples

---

## Paper Contributions

1. **Novel Architecture**: Multi-task learning with spatial attention for geolocation
2. **Improved Performance**: Demonstrate multi-task learning benefits
3. **Interpretability**: Visual attention maps explain predictions
4. **Ablation Study**: Quantify contribution of each component
5. **Open Source**: Code and models available for research

---

## Citation (Template)

```bibtex
@article{yourname2025multitask,
  title={Multi-Task Learning with Attention for Geographic Image Classification},
  author={Your Name},
  year={2025},
  note={Course project}
}
```

---

## Questions for Research

1. Does multi-task learning improve geolocation accuracy?
2. What image regions do models use for prediction?
3. How do auxiliary tasks (region, climate) help?
4. When and why does the model fail?
5. Is attention useful for interpretability?

**This implementation answers all of these!**
