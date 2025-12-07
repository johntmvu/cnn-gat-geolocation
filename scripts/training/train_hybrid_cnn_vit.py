"""
Hybrid CNN-ViT Multi-Task Model for Geolocation
Combines CNN for local features + CLIP Vision Transformer for global context
Uses pretrained CLIP ViT for superior feature extraction
Novel architecture for geographic image classification
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

# Dataset (reuse from train_multitask)
class MultiTaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        print("Loading multi-task dataset...")
        countries = sorted([c for c in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, c)) and not c.startswith('.')])
        
        self.country_to_idx = {country: idx for idx, country in enumerate(countries)}
        
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
        
        loss_country = country_criterion(country_out, country_labels)
        loss_region = region_criterion(region_out, region_labels)
        loss_climate = climate_criterion(climate_out, climate_labels)
        
        loss = loss_country + 0.3 * loss_region + 0.3 * loss_climate
        
        loss.backward()
        optimizer.step()
        
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
    print("\n" + "="*60)
    print("HYBRID CNN-ViT MULTI-TASK TRAINING")
    print("="*60)
    
    # Load dataset
    dataset = MultiTaskDataset(DATA_DIR, transform=train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Model
    num_countries = len(dataset.country_to_idx)
    num_regions = len(dataset.region_to_idx)
    num_climates = len(dataset.climate_to_idx)
    
    model = HybridCNNViT(
        num_countries=num_countries,
        num_regions=num_regions,
        num_climates=num_climates,
        clip_model_name=CLIP_MODEL
    )
    model = model.to(DEVICE)
    
    print(f"\nModel: Hybrid CNN-CLIP ViT")
    print(f"  CNN Backbone: ResNet50")
    print(f"  ViT Backbone: {CLIP_MODEL} (pretrained)")
    print(f"  Countries: {num_countries}")
    print(f"  Regions: {num_regions}")
    print(f"  Climates: {num_climates}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params/1e6:.1f}M")
    print(f"  Trainable Parameters: {trainable_params/1e6:.1f}M")
    print(f"  CLIP Status: Frozen (fine-tune after epoch 5)")
    
    # Loss and optimizer
    criteria = (nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Training loop
    best_country_acc = 0.0
    metrics = {
        "train_loss": [], "val_loss": [],
        "train_country_acc": [], "val_country_acc": [],
        "train_region_acc": [], "val_region_acc": [],
        "train_climate_acc": [], "val_climate_acc": []
    }
    
    print("\nStarting training...\n")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Unfreeze CLIP after epoch 5 for fine-tuning
        if epoch == 5:
            print("\n🔓 Unfreezing CLIP for fine-tuning...")
            model.unfreeze_clip()
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * 0.1
            print(f"Reduced learning rate to {LEARNING_RATE * 0.1:.6f}\n")
        
        train_loss, train_country, train_region, train_climate = train_epoch(
            model, train_loader, criteria, optimizer)
        val_loss, val_country, val_region, val_climate = validate(
            model, val_loader, criteria)
        
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
        
        if val_country > best_country_acc:
            best_country_acc = val_country
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_hybrid_cnn_vit.pth"))
            print(f"✓ Saved best model (Country: {val_country:.2f}%)")
        
        print()
    
    # Save mappings and metrics
    mappings = {
        "country_to_idx": dataset.country_to_idx,
        "region_to_idx": dataset.region_to_idx,
        "climate_to_idx": dataset.climate_to_idx
    }
    
    with open(os.path.join(MODEL_DIR, "hybrid_cnn_vit_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)
    
    with open(os.path.join(MODEL_DIR, "hybrid_cnn_vit_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Country Accuracy: {best_country_acc:.2f}%")
    print(f"Model: models/best_hybrid_cnn_vit.pth")
    print("="*60)
