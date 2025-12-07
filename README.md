# CNN-GAT Geolocation

Multi-task deep learning models for geographic image classification using street view data.

## 📁 Project Structure

```
cnn-gat-geolocation/
├── scripts/
│   ├── training/           # Model training scripts
│   │   ├── train_country_model.py      # Baseline ResNet50
│   │   ├── train_multitask.py          # Multi-task with attention
│   │   ├── train_hybrid_cnn_vit.py     # Hybrid CNN + CLIP
│   │   └── train_clip_only.py          # CLIP-only model
│   ├── evaluation/         # Model comparison and testing
│   │   ├── compare_models.py           # Baseline vs Multi-task
│   │   ├── compare_all_models.py       # All 4 models comparison
│   │   └── test_random_images.py       # Random sample testing
│   └── visualization/      # Attention and results visualization
│       └── visualize_attention.py
├── models/                 # Saved model checkpoints and metrics
├── data/                   # Dataset directory
│   └── gsv_50k/           # Google Street View 50K dataset
├── outputs/               # Generated outputs
│   ├── attention_maps/    # Attention visualizations
│   └── metrics/           # Training curves and comparisons
├── docs/                  # Documentation
│   ├── README_RESEARCH.md
│   └── PRESENTATION_OUTLINE.md
├── archive/               # Deprecated scripts
└── env.yml               # Conda environment file

```

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda env create -f env.yml
conda activate geolocation
```

### 2. Train Models
```bash
# Baseline CNN
python scripts/training/train_country_model.py

# Multi-task CNN with Attention
python scripts/training/train_multitask.py

# Hybrid CNN-CLIP (requires CLIP installation)
python scripts/training/train_hybrid_cnn_vit.py

# CLIP-Only (requires CLIP installation)
python scripts/training/train_clip_only.py
```

### 3. Compare Models
```bash
# Compare all trained models
python scripts/evaluation/compare_all_models.py

# Test on random samples
python scripts/evaluation/test_random_images.py
```

### 4. Visualize Attention
```bash
python scripts/visualization/visualize_attention.py
```

## 🏗️ Model Architectures

### 1. Baseline CNN (ResNet50)
- Single-task country classification
- Pre-trained on ImageNet
- 25.6M parameters
- Fast training baseline

### 2. Multi-Task CNN
- Hierarchical prediction: Country + Region + Climate
- Spatial attention mechanism
- Multi-task learning regularization
- 25.8M parameters

### 3. Hybrid CNN-CLIP
- ResNet50 for local features
- CLIP ViT for global semantic context
- Feature fusion with cross-attention
- 180M parameters (28.7M trainable initially)

### 4. CLIP-Only
- Pure vision transformer approach
- Pre-trained on 400M image-text pairs
- Minimal trainable parameters (2.5M)
- Best parameter efficiency

## 📊 Dataset

**GSV_50K**: 50,000 street view images across 120+ countries
- Training: 40,000 images (80%)
- Validation: 10,000 images (20%)
- Organized by country folders
- Geographic metadata: region, climate

## 🎯 Tasks

1. **Country Classification**: 120+ countries
2. **Region Classification**: 7 geographic regions
3. **Climate Classification**: 6 climate zones

## 📈 Results

Model performance and comparisons available in `outputs/metrics/`

## 📚 Documentation

- **Research Workflow**: `docs/README_RESEARCH.md`
- **Presentation Outline**: `docs/PRESENTATION_OUTLINE.md`

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- See `env.yml` for complete dependencies

## 📝 Citation

If you use this code for research, please cite appropriately.

## 📄 License

MIT License
