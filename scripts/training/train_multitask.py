"""
Multi-Task Learning with Attention for Geolocation
Predicts Country + Region + Climate with attention mechanism for interpretability
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
import sys
from pathlib import Path
# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/gsv_50k/compressed_dataset"
MODEL_DIR = PROJECT_ROOT / "models"
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4

os.makedirs(MODEL_DIR, exist_ok=True)
DATA_DIR = str(DATA_DIR)
MODEL_DIR = str(MODEL_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define geographic regions and climate zones
REGION_MAPPING = {
    # North America
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "Guatemala": "North America", "Costa Rica": "North America", "Puerto Rico": "North America",
    "Dominican Republic": "North America", "Bermuda": "North America", "Greenland": "North America",
    "Curacao": "North America", "US Virgin Islands": "North America",
    
    # South America
    "Brazil": "South America", "Argentina": "South America", "Chile": "South America",
    "Colombia": "South America", "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Bolivia": "South America", "Uruguay": "South America",
    "Paraguay": "South America",
    
    # Europe
    "United Kingdom": "Europe", "France": "Europe", "Germany": "Europe", "Italy": "Europe",
    "Spain": "Europe", "Poland": "Europe", "Ukraine": "Europe", "Romania": "Europe",
    "Netherlands": "Europe", "Belgium": "Europe", "Greece": "Europe", "Portugal": "Europe",
    "Sweden": "Europe", "Austria": "Europe", "Switzerland": "Europe", "Denmark": "Europe",
    "Finland": "Europe", "Norway": "Europe", "Ireland": "Europe", "Croatia": "Europe",
    "Bulgaria": "Europe", "Serbia": "Europe", "Slovakia": "Europe", "Lithuania": "Europe",
    "Slovenia": "Europe", "Latvia": "Europe", "Estonia": "Europe", "Montenegro": "Europe",
    "Luxembourg": "Europe", "Malta": "Europe", "Iceland": "Europe", "Russia": "Europe",
    "Czechia": "Europe", "Hungary": "Europe", "Belarus": "Europe", "Monaco": "Europe",
    "Faroe Islands": "Europe", "Gibraltar": "Europe", "Isle of Man": "Europe",
    "Jersey": "Europe", "San Marino": "Europe", "North Macedonia": "Europe",
    "Svalbard and Jan Mayen": "Europe", "Aland": "Europe",
    
    # Asia
    "China": "Asia", "India": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia", "Malaysia": "Asia",
    "Indonesia": "Asia", "Turkey": "Asia", "Israel": "Asia", "Singapore": "Asia",
    "Hong Kong": "Asia", "Taiwan": "Asia", "UAE": "Asia", "United Arab Emirates": "Asia",
    "Jordan": "Asia", "Lebanon": "Asia", "Qatar": "Asia", "Sri Lanka": "Asia",
    "Pakistan": "Asia", "Bangladesh": "Asia", "Nepal": "Asia", "Bhutan": "Asia",
    "Cambodia": "Asia", "Laos": "Asia", "Myanmar": "Asia", "Mongolia": "Asia",
    "Kyrgyzstan": "Asia", "Palestine": "Asia", "Iraq": "Asia", "Macao": "Asia",
    "Guam": "Asia", "Northern Mariana Islands": "Asia",
    
    # Africa
    "South Africa": "Africa", "Egypt": "Africa", "Kenya": "Africa", "Nigeria": "Africa",
    "Ghana": "Africa", "Tunisia": "Africa", "Senegal": "Africa", "Uganda": "Africa",
    "Tanzania": "Africa", "Mozambique": "Africa", "Botswana": "Africa", "Lesotho": "Africa",
    "Eswatini": "Africa", "Madagascar": "Africa", "Reunion": "Africa", "South Sudan": "Africa",
    
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania", "American Samoa": "Oceania",
    "Pitcairn Islands": "Oceania", "South Georgia and South Sandwich Islands": "Oceania",
    
    # Middle East (can also categorize as Asia)
    "Armenia": "Asia", "Martinique": "North America", "Andorra": "Europe",
    "Antarctica": "Antarctica"
}

CLIMATE_MAPPING = {
    # Tropical
    "Brazil": "Tropical", "Indonesia": "Tropical", "Malaysia": "Tropical",
    "Philippines": "Tropical", "Thailand": "Tropical", "Vietnam": "Tropical",
    "Singapore": "Tropical", "Sri Lanka": "Tropical", "Cambodia": "Tropical",
    "Myanmar": "Tropical", "Laos": "Tropical", "Madagascar": "Tropical",
    "American Samoa": "Tropical", "Guam": "Tropical",
    
    # Temperate
    "United States": "Temperate", "China": "Temperate", "Japan": "Temperate",
    "South Korea": "Temperate", "United Kingdom": "Temperate", "France": "Temperate",
    "Germany": "Temperate", "Italy": "Temperate", "Spain": "Temperate",
    "Portugal": "Temperate", "Australia": "Temperate", "New Zealand": "Temperate",
    "Argentina": "Temperate", "Chile": "Temperate", "Uruguay": "Temperate",
    
    # Continental
    "Canada": "Continental", "Russia": "Continental", "Poland": "Continental",
    "Ukraine": "Continental", "Belarus": "Continental", "Kazakhstan": "Continental",
    "Mongolia": "Continental", "Czechia": "Continental", "Slovakia": "Continental",
    "Hungary": "Continental", "Romania": "Continental", "Bulgaria": "Continental",
    
    # Arid/Desert
    "Egypt": "Arid", "UAE": "Arid", "United Arab Emirates": "Arid",
    "Qatar": "Arid", "Jordan": "Arid", "Israel": "Arid", "Iraq": "Arid",
    "Botswana": "Arid", "Namibia": "Arid", "Mongolia": "Arid",
    
    # Mediterranean
    "Greece": "Mediterranean", "Turkey": "Mediterranean", "Lebanon": "Mediterranean",
    "Tunisia": "Mediterranean", "Morocco": "Mediterranean", "Croatia": "Mediterranean",
    "Malta": "Mediterranean", "Monaco": "Mediterranean", "San Marino": "Mediterranean",
    
    # Polar/Cold
    "Iceland": "Polar", "Norway": "Polar", "Sweden": "Polar", "Finland": "Polar",
    "Greenland": "Polar", "Antarctica": "Polar", "Svalbard and Jan Mayen": "Polar",
    "Faroe Islands": "Polar"
}

# Fill in missing countries with reasonable defaults
def get_region(country):
    return REGION_MAPPING.get(country, "Other")

def get_climate(country):
    return CLIMATE_MAPPING.get(country, "Temperate")

# Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention, attention

# Multi-Task Model with Attention
class MultiTaskGeoModel(nn.Module):
    def __init__(self, num_countries, num_regions, num_climates):
        super(MultiTaskGeoModel, self).__init__()
        
        # Backbone: ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Attention layer
        self.attention = SpatialAttention(2048)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-task heads
        self.country_head = nn.Linear(2048, num_countries)
        self.region_head = nn.Linear(2048, num_regions)
        self.climate_head = nn.Linear(2048, num_climates)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attended_features, attention_map = self.attention(features)
        
        # Global pooling
        pooled = self.avgpool(attended_features)
        pooled = torch.flatten(pooled, 1)
        
        # Multi-task predictions
        country_out = self.country_head(pooled)
        region_out = self.region_head(pooled)
        climate_out = self.climate_head(pooled)
        
        return country_out, region_out, climate_out, attention_map

# Dataset with multiple labels
class MultiTaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        print("Loading multi-task dataset...")
        countries = sorted([c for c in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, c)) and not c.startswith('.')])
        
        self.country_to_idx = {country: idx for idx, country in enumerate(countries)}
        
        # Get unique regions and climates
        regions = sorted(set(get_region(c) for c in countries))
        climates = sorted(set(get_climate(c) for c in countries))
        
        self.region_to_idx = {region: idx for idx, region in enumerate(regions)}
        self.climate_to_idx = {climate: idx for idx, climate in enumerate(climates)}
        
        print(f"Countries: {len(countries)}, Regions: {len(regions)}, Climates: {len(climates)}")
        
        for country in tqdm(countries, desc="Loading images"):
            country_path = os.path.join(data_dir, country)
            country_idx = self.country_to_idx[country]
            region_idx = self.region_to_idx[get_region(country)]
            climate_idx = self.climate_to_idx[get_climate(country)]
            
            for img_name in os.listdir(country_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(country_path, img_name),
                        country_idx,
                        region_idx,
                        climate_idx
                    ))
        
        print(f"Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, country_idx, region_idx, climate_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, country_idx, region_idx, climate_idx
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training functions
def train_epoch(model, loader, criteria, optimizer):
    model.train()
    running_loss = 0.0
    country_correct = 0
    region_correct = 0
    climate_correct = 0
    total = 0
    
    country_criterion, region_criterion, climate_criterion = criteria
    
    for images, country_labels, region_labels, climate_labels in tqdm(loader, desc="Training"):
        images = images.to(DEVICE)
        country_labels = country_labels.to(DEVICE)
        region_labels = region_labels.to(DEVICE)
        climate_labels = climate_labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        country_out, region_out, climate_out, _ = model(images)
        
        # Multi-task loss
        loss_country = country_criterion(country_out, country_labels)
        loss_region = region_criterion(region_out, region_labels)
        loss_climate = climate_criterion(climate_out, climate_labels)
        
        # Weighted combination
        loss = loss_country + 0.3 * loss_region + 0.3 * loss_climate
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracies
        _, country_pred = country_out.max(1)
        _, region_pred = region_out.max(1)
        _, climate_pred = climate_out.max(1)
        
        total += country_labels.size(0)
        country_correct += country_pred.eq(country_labels).sum().item()
        region_correct += region_pred.eq(region_labels).sum().item()
        climate_correct += climate_pred.eq(climate_labels).sum().item()
    
    return (running_loss / len(loader),
            100. * country_correct / total,
            100. * region_correct / total,
            100. * climate_correct / total)

def validate(model, loader, criteria):
    model.eval()
    running_loss = 0.0
    country_correct = 0
    region_correct = 0
    climate_correct = 0
    total = 0
    
    country_criterion, region_criterion, climate_criterion = criteria
    
    with torch.no_grad():
        for images, country_labels, region_labels, climate_labels in tqdm(loader, desc="Validation"):
            images = images.to(DEVICE)
            country_labels = country_labels.to(DEVICE)
            region_labels = region_labels.to(DEVICE)
            climate_labels = climate_labels.to(DEVICE)
            
            country_out, region_out, climate_out, _ = model(images)
            
            loss_country = country_criterion(country_out, country_labels)
            loss_region = region_criterion(region_out, region_labels)
            loss_climate = climate_criterion(climate_out, climate_labels)
            
            loss = loss_country + 0.3 * loss_region + 0.3 * loss_climate
            
            running_loss += loss.item()
            
            _, country_pred = country_out.max(1)
            _, region_pred = region_out.max(1)
            _, climate_pred = climate_out.max(1)
            
            total += country_labels.size(0)
            country_correct += country_pred.eq(country_labels).sum().item()
            region_correct += region_pred.eq(region_labels).sum().item()
            climate_correct += climate_pred.eq(climate_labels).sum().item()
    
    return (running_loss / len(loader),
            100. * country_correct / total,
            100. * region_correct / total,
            100. * climate_correct / total)

# Main training
if __name__ == '__main__':
    # Load dataset
    dataset = MultiTaskDataset(DATA_DIR, transform=train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"\nTraining samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Model
    num_countries = len(dataset.country_to_idx)
    num_regions = len(dataset.region_to_idx)
    num_climates = len(dataset.climate_to_idx)
    
    model = MultiTaskGeoModel(num_countries, num_regions, num_climates)
    model = model.to(DEVICE)
    
    print(f"\nModel: Countries={num_countries}, Regions={num_regions}, Climates={num_climates}")
    
    # Loss functions and optimizer
    criteria = (nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_country_acc = 0.0
    metrics = {
        "train_loss": [], "val_loss": [],
        "train_country_acc": [], "val_country_acc": [],
        "train_region_acc": [], "val_region_acc": [],
        "train_climate_acc": [], "val_climate_acc": []
    }
    
    print("\nStarting multi-task training...\n")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        train_loss, train_country, train_region, train_climate = train_epoch(
            model, train_loader, criteria, optimizer)
        val_loss, val_country, val_region, val_climate = validate(
            model, val_loader, criteria)
        
        # Log metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_country_acc"].append(train_country)
        metrics["val_country_acc"].append(val_country)
        metrics["train_region_acc"].append(train_region)
        metrics["val_region_acc"].append(val_region)
        metrics["train_climate_acc"].append(train_climate)
        metrics["val_climate_acc"].append(val_climate)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"  Country: {train_country:.2f}% | Region: {train_region:.2f}% | Climate: {train_climate:.2f}%")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"  Country: {val_country:.2f}% | Region: {val_region:.2f}% | Climate: {val_climate:.2f}%")
        
        # Save best model
        if val_country > best_country_acc:
            best_country_acc = val_country
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_multitask_model.pth"))
            print(f"✓ Saved best model (Country Acc: {val_country:.2f}%)")
        
        print()
    
    # Save mappings and metrics
    mappings = {
        "country_to_idx": dataset.country_to_idx,
        "region_to_idx": dataset.region_to_idx,
        "climate_to_idx": dataset.climate_to_idx
    }
    
    with open(os.path.join(MODEL_DIR, "multitask_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)
    
    with open(os.path.join(MODEL_DIR, "multitask_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Country Accuracy: {best_country_acc:.2f}%")
    print(f"Model saved to: {os.path.join(MODEL_DIR, 'best_multitask_model.pth')}")
