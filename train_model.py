"""
train_model.py — Production-style training for Google Landmarks Dataset (ResNet50)
Author: John Vu

This script:
- Loads metadata from gldv2_dataset/train/train_clean.csv
- Builds DataLoaders from nested /train/0/0/0/... image paths
- Fine-tunes a pretrained ResNet50
- Saves best model and logs to /models and /outputs
"""

import os
import json
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_TRAIN_CSV = "gldv2_micro/train.csv"
DATASET_VAL_CSV = "gldv2_micro/val.csv"
IMAGE_ROOT = "gldv2_micro/images"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 0

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# DATASET
# ============================================================

class LandmarkDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.samples = list(zip(df["filename"], df["landmark_id"]))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]

        # Build the full image path without nested directories
        img_path = os.path.join(self.root_dir, f"{img_name}.jpg")

        from PIL import Image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self.samples))  # Get next sample

        if self.transform:
            image = self.transform(image)

        return image, label



# ============================================================
# DATA PREPARATION
# ============================================================

print("Loading CSV metadata...")
# Directly load pre-made training and validation sets
train_df = pd.read_csv(DATASET_TRAIN_CSV)
val_df = pd.read_csv(DATASET_VAL_CSV)

# Drop rows with missing values in train and validation separately
train_df = train_df.dropna(subset=["filename", "landmark_id"])
val_df = val_df.dropna(subset=["filename", "landmark_id"])

# Ensure landmark_id is an integer in both datasets
train_df["landmark_id"] = train_df["landmark_id"].astype(int)
val_df["landmark_id"] = val_df["landmark_id"].astype(int)

# Map landmark_id to sequential indices for train and validation separately
print("Mapping landmark_id to sequential indices for train and validation...")
label2id = {label: idx for idx, label in enumerate(sorted(train_df["landmark_id"].unique()))}
id2label = {v: k for k, v in label2id.items()}
train_df["landmark_id"] = train_df["landmark_id"].map(label2id)
val_df["landmark_id"] = val_df["landmark_id"].map(label2id)

num_classes = len(label2id)
print(f"Number of classes: {num_classes}")

train_df["label"] = train_df["landmark_id"].map(label2id)
val_df["label"] = val_df["landmark_id"].map(label2id)

train_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = LandmarkDataset(train_df, IMAGE_ROOT, train_tfms)
val_ds = LandmarkDataset(val_df, IMAGE_ROOT, val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ============================================================
# MODEL SETUP
# ============================================================

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss / total, correct / total


def validate_one_epoch(epoch):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss / total, correct / total


best_val_acc = 0.0
metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate_one_epoch(epoch)
        scheduler.step(val_loss)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_resnet50.pth"))
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")

# Save metrics
json.dump(metrics, open(os.path.join(OUTPUT_DIR, "metrics", "training_metrics.json"), "w"), indent=2)
print("Training complete.")
