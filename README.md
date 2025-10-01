# Geolocation CNN (Google Landmarks Subset)

This project trains a Convolutional Neural Network (CNN) to recognize landmarks from the Google Landmarks Dataset v2 (GLDv2).

## Repository Structure

- `data/` → raw + processed images
- `dataset_creation/` → scripts for downloading + preprocessing
- `models/` → CNN architectures
- `training/` → training loops
- `evaluation/` → evaluation scripts
- `runs/` → logs
- `saved_models/` → model checkpoints
- `config.py` → hyperparameters and paths

## Quickstart

```bash
# Clone repo
git clone <repo-url>
cd geolocation-cnn

# Setup environment
conda env create -f env.yml
conda activate geolocation

# Download a small subset for testing
python dataset_creation/subset_gldv2.py
