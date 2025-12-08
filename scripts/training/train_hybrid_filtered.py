"""
Hybrid CNN-ViT Multi-Task Model for Geolocation - FILTERED DATASET
Trains only on countries with at least 10 images to reduce class imbalance
Combines CNN for local features + CLIP Vision Transformer for global context
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
import math
import clip

# Configuration
import sys
from pathlib import Path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = str(PROJECT_ROOT / "data/gsv_50k/compressed_dataset")
MODEL_DIR = str(PROJECT_ROOT / "models")
MIN_IMAGES_PER_COUNTRY = 10  # Filter threshold
BATCH_SIZE = 64  # Smaller due to ViT memory requirements
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4
CLIP_MODEL = "ViT-B/32"  # CLIP pretrained model

os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Using CLIP model: {CLIP_MODEL}")
print(f"Filtering: Countries with < {MIN_IMAGES_PER_COUNTRY} images will be excluded")

# Import region and climate mappings from train_multitask
from scripts.training.train_multitask import REGION_MAPPING, CLIMATE_MAPPING, get_region, get_climate

# Adapter layer to bridge CNN and CLIP features
class FeatureFusionModule(nn.Module):
    """Fuses CNN local features with CLIP global features"""
    def __init__(self, cnn_dim=2048, clip_dim=512, output_dim=768):
        super().__init__()
        
        # Project CNN features
        self.cnn_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cnn_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Project CLIP features
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Fusion attention
        self.fusion_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, cnn_features, clip_features):
        # Convert CLIP features to float32 if needed
        if clip_features.dtype != cnn_features.dtype:
            clip_features = clip_features.float()
        
        # Project both feature types
        cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)  # (batch, 1, output_dim)
        clip_proj = self.clip_proj(clip_features).unsqueeze(1)  # (batch, 1, output_dim)
        
        # Concatenate for cross-attention
        combined = torch.cat([cnn_proj, clip_proj], dim=1)  # (batch, 2, output_dim)
        
        # Self-attention to fuse features
        fused, attn_weights = self.fusion_attn(combined, combined, combined)
        fused = self.norm(fused)
        
        # Return mean of fused features
        return fused.mean(dim=1), attn_weights  # (batch, output_dim)

# Hybrid CNN-CLIP Model
class HybridCNNViT(nn.Module):
    """
    Hybrid architecture combining CNN and CLIP Vision Transformer
    
    CNN (ResNet50): Extract local features (edges, textures, objects)
    CLIP ViT: Capture global semantic context with pretrained knowledge
    Fusion Module: Intelligently combines both feature streams
    """
    def __init__(self, num_countries, num_regions, num_climates, 
                 clip_model_name="ViT-B/32"):
        super().__init__()
        
        # CNN Backbone: ResNet50 (without final pooling and FC)
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output: (batch, 2048, 7, 7)
        
        # CLIP Vision Transformer (pretrained)
        self.clip_model, _ = clip.load(clip_model_name, device=DEVICE)
        # Freeze CLIP initially (can fine-tune later)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # CLIP visual output dimension
        clip_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/32
        
        # Feature fusion module
        self.fusion = FeatureFusionModule(
            cnn_dim=2048, 
            clip_dim=clip_dim, 
            output_dim=768
        )
        
        # Multi-task heads
        self.country_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_countries)
        )
        
        self.region_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_regions)
        )
        
        self.climate_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_climates)
        )
        
    def forward(self, x):
        # CNN feature extraction (local features)
        cnn_features = self.cnn_backbone(x)  # (batch, 2048, 7, 7)
        
        # CLIP feature extraction (global semantic features)
        with torch.no_grad():  # CLIP frozen
            clip_features = self.clip_model.encode_image(x)  # (batch, 512)
        
        # Fuse CNN and CLIP features
        fused_features, fusion_attn = self.fusion(cnn_features, clip_features)
        # (batch, 768)
        
        # Multi-task predictions
        country_out = self.country_head(fused_features)
        region_out = self.region_head(fused_features)
        climate_out = self.climate_head(fused_features)
        
        # Return fusion attention for visualization
        return country_out, region_out, climate_out, fusion_attn
    
    def unfreeze_clip(self):
        """Unfreeze CLIP for fine-tuning after initial training"""
        for param in self.clip_model.parameters():
            param.requires_grad = True
        print("CLIP ViT unfrozen for fine-tuning")

# Filtered Dataset
class FilteredMultiTaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, min_images=10):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.min_images = min_images
        
        print(f"Loading FILTERED multi-task dataset (min {min_images} images per country)...")
        
        # First pass: count images per country
        all_countries = sorted([c for c in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, c)) and not c.startswith('.')])
        
        country_image_counts = {}
        for country in all_countries:
            country_path = os.path.join(data_dir, country)
            image_files = [f for f in os.listdir(country_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            country_image_counts[country] = len(image_files)
        
        # Filter countries with sufficient images
        filtered_countries = [c for c, count in country_image_counts.items() 
                            if count >= min_images]
        excluded_countries = [c for c, count in country_image_counts.items() 
                             if count < min_images]
        
        print(f"Original countries: {len(all_countries)}")
        print(f"Filtered countries: {len(filtered_countries)} (kept)")
        print(f"Excluded countries: {len(excluded_countries)} (< {min_images} images)")
        
        if excluded_countries:
            print(f"Excluded: {', '.join(excluded_countries[:10])}" + 
                  (f" and {len(excluded_countries)-10} more" if len(excluded_countries) > 10 else ""))
        
        # Create mappings with filtered countries
        self.country_to_idx = {country: idx for idx, country in enumerate(sorted(filtered_countries))}
        
        regions = sorted(set(get_region(c) for c in filtered_countries))
        climates = sorted(set(get_climate(c) for c in filtered_countries))
        
        self.region_to_idx = {region: idx for idx, region in enumerate(regions)}
        self.climate_to_idx = {climate: idx for idx, climate in enumerate(climates)}
        
        print(f"Classes: {len(filtered_countries)} countries, {len(regions)} regions, {len(climates)} climates")
        
        # Load images from filtered countries
        for country in tqdm(sorted(filtered_countries), desc="Loading images"):
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
        
        print(f"Loaded {len(self.samples)} images from {len(filtered_countries)} countries")
        
        # Print statistics
        total_original = sum(country_image_counts.values())
        print(f"Dataset size: {len(self.samples)}/{total_original} images " + 
              f"({100*len(self.samples)/total_original:.1f}% retained)")
    
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
        
        loss_country = country_criterion(country_out, country_labels)
        loss_region = region_criterion(region_out, region_labels)
        loss_climate = climate_criterion(climate_out, climate_labels)
        
        # Weighted multi-task loss
        loss = loss_country + 0.3 * loss_region + 0.3 * loss_climate
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, country_pred = torch.max(country_out, 1)
        _, region_pred = torch.max(region_out, 1)
        _, climate_pred = torch.max(climate_out, 1)
        
        country_correct += (country_pred == country_labels).sum().item()
        region_correct += (region_pred == region_labels).sum().item()
        climate_correct += (climate_pred == climate_labels).sum().item()
        total += country_labels.size(0)
    
    epoch_loss = running_loss / len(loader)
    country_acc = 100.0 * country_correct / total
    region_acc = 100.0 * region_correct / total
    climate_acc = 100.0 * climate_correct / total
    
    return epoch_loss, country_acc, region_acc, climate_acc

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
            
            # Calculate accuracy
            _, country_pred = torch.max(country_out, 1)
            _, region_pred = torch.max(region_out, 1)
            _, climate_pred = torch.max(climate_out, 1)
            
            country_correct += (country_pred == country_labels).sum().item()
            region_correct += (region_pred == region_labels).sum().item()
            climate_correct += (climate_pred == climate_labels).sum().item()
            total += country_labels.size(0)
    
    val_loss = running_loss / len(loader)
    country_acc = 100.0 * country_correct / total
    region_acc = 100.0 * region_correct / total
    climate_acc = 100.0 * climate_correct / total
    
    return val_loss, country_acc, region_acc, climate_acc

def main():
    print("="*60)
    print("HYBRID CNN-CLIP TRAINING - FILTERED DATASET")
    print(f"Minimum images per country: {MIN_IMAGES_PER_COUNTRY}")
    print("="*60)
    
    # Load filtered dataset
    full_dataset = FilteredMultiTaskDataset(
        DATA_DIR, 
        transform=train_transform,
        min_images=MIN_IMAGES_PER_COUNTRY
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # Get number of classes from filtered dataset
    num_countries = len(full_dataset.country_to_idx)
    num_regions = len(full_dataset.region_to_idx)
    num_climates = len(full_dataset.climate_to_idx)
    
    print(f"\nModel configuration:")
    print(f"  Countries: {num_countries}")
    print(f"  Regions: {num_regions}")
    print(f"  Climates: {num_climates}")
    
    # Initialize model
    model = HybridCNNViT(num_countries, num_regions, num_climates, CLIP_MODEL)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    country_criterion = nn.CrossEntropyLoss()
    region_criterion = nn.CrossEntropyLoss()
    climate_criterion = nn.CrossEntropyLoss()
    criteria = (country_criterion, region_criterion, climate_criterion)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_val_acc = 0.0
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_country_acc': [], 'val_country_acc': [],
        'train_region_acc': [], 'val_region_acc': [],
        'train_climate_acc': [], 'val_climate_acc': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Unfreeze CLIP after first 5 epochs
        if epoch == 5:
            print("\n🔓 Unfreezing CLIP for fine-tuning...")
            model.unfreeze_clip()
            # Update optimizer to include CLIP parameters
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Train
        train_loss, train_country, train_region, train_climate = train_epoch(
            model, train_loader, criteria, optimizer
        )
        
        # Validate
        val_loss, val_country, val_region, val_climate = validate(
            model, val_loader, criteria
        )
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_country_acc'].append(train_country)
        metrics['val_country_acc'].append(val_country)
        metrics['train_region_acc'].append(train_region)
        metrics['val_region_acc'].append(val_region)
        metrics['train_climate_acc'].append(train_climate)
        metrics['val_climate_acc'].append(val_climate)
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Country: {train_country:.2f}% | Val Country: {val_country:.2f}%")
        print(f"  Train Region: {train_region:.2f}% | Val Region: {val_region:.2f}%")
        print(f"  Train Climate: {train_climate:.2f}% | Val Climate: {val_climate:.2f}%")
        
        # Save best model
        if val_country > best_val_acc:
            best_val_acc = val_country
            torch.save(model.state_dict(), 
                      os.path.join(MODEL_DIR, "best_hybrid_filtered.pth"))
            print(f"  ✓ New best model saved! (Country Acc: {val_country:.2f}%)")
    
    # Save final metrics and mappings
    with open(os.path.join(MODEL_DIR, "hybrid_filtered_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    mappings = {
        "country_to_idx": full_dataset.country_to_idx,
        "region_to_idx": full_dataset.region_to_idx,
        "climate_to_idx": full_dataset.climate_to_idx,
        "min_images_threshold": MIN_IMAGES_PER_COUNTRY,
        "num_countries": num_countries,
        "num_regions": num_regions,
        "num_climates": num_climates
    }
    with open(os.path.join(MODEL_DIR, "hybrid_filtered_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation country accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {MODEL_DIR}/best_hybrid_filtered.pth")
    print(f"Metrics saved: {MODEL_DIR}/hybrid_filtered_metrics.json")
    print(f"Mappings saved: {MODEL_DIR}/hybrid_filtered_mappings.json")
    print(f"\nFiltered dataset statistics:")
    print(f"  Countries included: {num_countries}")
    print(f"  Minimum images per country: {MIN_IMAGES_PER_COUNTRY}")
    print(f"  Total images: {len(full_dataset)}")

if __name__ == "__main__":
    main()
