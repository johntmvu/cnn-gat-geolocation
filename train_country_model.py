"""
Simple Country Classification Model
Trains a ResNet50 to predict countries from street view images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import json

# Configuration
DATA_DIR = "data/gsv_50k/compressed_dataset"  # Original dataset
MODEL_DIR = "models"
BATCH_SIZE = 128  # Larger batch for A100
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4  # Increase workers

os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset
class CountryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.countries = sorted(os.listdir(data_dir))
        self.countries = [c for c in self.countries if os.path.isdir(os.path.join(data_dir, c)) and not c.startswith('.')]
        self.country_to_idx = {country: idx for idx, country in enumerate(self.countries)}
        
        # Load all image paths
        print("Loading dataset...")
        for country in tqdm(self.countries):
            country_path = os.path.join(data_dir, country)
            for img_name in os.listdir(country_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(country_path, img_name), self.country_to_idx[country]))
        
        print(f"Found {len(self.samples)} images across {len(self.countries)} countries")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Main training code
if __name__ == '__main__':
    # Load dataset
    dataset = CountryDataset(DATA_DIR, transform=train_transform)

    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Model
    num_classes = len(dataset.countries)
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_acc = 0.0
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_country_model.pth"))
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    # Save final metrics and country mapping
    with open(os.path.join(MODEL_DIR, "country_mapping.json"), "w") as f:
        json.dump(dataset.country_to_idx, f, indent=2)

    with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(MODEL_DIR, 'best_country_model.pth')}")
