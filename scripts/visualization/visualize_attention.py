"""
Attention Visualization for Multi-Task Geolocation Model
Generates heatmaps showing which image regions the model focuses on
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Import model architecture
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_multitask import MultiTaskGeoModel

# Configuration
MODEL_PATH = str(PROJECT_ROOT / "models/best_multitask_model.pth")
MAPPINGS_PATH = str(PROJECT_ROOT / "models/multitask_mappings.json")
DATA_DIR = str(PROJECT_ROOT / "data/gsv_50k/compressed_dataset")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs/attention_maps")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# Load mappings
with open(MAPPINGS_PATH, "r") as f:
    mappings = json.load(f)

country_to_idx = mappings["country_to_idx"]
region_to_idx = mappings["region_to_idx"]
climate_to_idx = mappings["climate_to_idx"]

idx_to_country = {v: k for k, v in country_to_idx.items()}
idx_to_region = {v: k for k, v in region_to_idx.items()}
idx_to_climate = {v: k for k, v in climate_to_idx.items()}

# Load model
num_countries = len(country_to_idx)
num_regions = len(region_to_idx)
num_climates = len(climate_to_idx)

model = MultiTaskGeoModel(num_countries, num_regions, num_climates)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"Model loaded: {num_countries} countries, {num_regions} regions, {num_climates} climates")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inverse normalization for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def predict_with_attention(image_path):
    """Predict and return attention map"""
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        country_out, region_out, climate_out, attention_map = model(image_tensor)
        
        # Get predictions
        country_probs = torch.nn.functional.softmax(country_out, dim=1)
        region_probs = torch.nn.functional.softmax(region_out, dim=1)
        climate_probs = torch.nn.functional.softmax(climate_out, dim=1)
        
        country_conf, country_idx = torch.max(country_probs, 1)
        region_conf, region_idx = torch.max(region_probs, 1)
        climate_conf, climate_idx = torch.max(climate_probs, 1)
        
        country = idx_to_country[country_idx.item()]
        region = idx_to_region[region_idx.item()]
        climate = idx_to_climate[climate_idx.item()]
        
        # Process attention map
        attention = attention_map[0, 0].cpu().numpy()  # Shape: (H, W)
        attention = cv2.resize(attention, (IMG_SIZE, IMG_SIZE))
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    return {
        'country': (country, country_conf.item() * 100),
        'region': (region, region_conf.item() * 100),
        'climate': (climate, climate_conf.item() * 100),
        'attention': attention,
        'image': original_image
    }

def create_attention_heatmap(image, attention_map, alpha=0.5):
    """Overlay attention heatmap on image"""
    # Convert image to numpy array
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
    
    return overlay

def visualize_single_prediction(image_path, save_path=None):
    """Visualize prediction with attention for a single image"""
    result = predict_with_attention(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(result['image'])
    axes[0].axis('off')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    
    # Attention heatmap
    heatmap = create_attention_heatmap(result['image'], result['attention'])
    axes[1].imshow(heatmap)
    axes[1].axis('off')
    axes[1].set_title('Attention Map\n(Where the model looks)', 
                     fontsize=12, fontweight='bold')
    
    # Attention only
    axes[2].imshow(result['attention'], cmap='jet')
    axes[2].axis('off')
    axes[2].set_title('Attention Weights', fontsize=12, fontweight='bold')
    
    # Add predictions as suptitle
    country, country_conf = result['country']
    region, region_conf = result['region']
    climate, climate_conf = result['climate']
    
    title = f"Predictions:\n"
    title += f"Country: {country} ({country_conf:.1f}%) | "
    title += f"Region: {region} ({region_conf:.1f}%) | "
    title += f"Climate: {climate} ({climate_conf:.1f}%)"
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()
    
    return result

def visualize_multiple_samples(num_samples=6):
    """Create a grid of attention visualizations from random samples"""
    import random
    
    # Get random images
    countries = [c for c in os.listdir(DATA_DIR) 
                if os.path.isdir(os.path.join(DATA_DIR, c)) and not c.startswith('.')]
    
    samples = []
    for _ in range(num_samples):
        country = random.choice(countries)
        country_path = os.path.join(DATA_DIR, country)
        images = [f for f in os.listdir(country_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            img_path = os.path.join(country_path, random.choice(images))
            samples.append((img_path, country))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for idx, (img_path, true_country) in enumerate(samples):
        result = predict_with_attention(img_path)
        pred_country, conf = result['country']
        
        # Original image
        axes[idx, 0].imshow(result['image'])
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        
        # Attention overlay
        heatmap = create_attention_heatmap(result['image'], result['attention'])
        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title('Attention Overlay', fontsize=12, fontweight='bold')
        
        # Predictions
        axes[idx, 2].axis('off')
        pred_text = f"True Country: {true_country}\n\n"
        pred_text += f"Predicted Country:\n{pred_country}\n({conf:.1f}%)\n\n"
        pred_text += f"Region: {result['region'][0]}\n({result['region'][1]:.1f}%)\n\n"
        pred_text += f"Climate: {result['climate'][0]}\n({result['climate'][1]:.1f}%)"
        
        color = 'green' if pred_country == true_country else 'red'
        axes[idx, 2].text(0.1, 0.5, pred_text, fontsize=11, 
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        if idx == 0:
            axes[idx, 2].set_title('Predictions', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "attention_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid visualization to: {save_path}")
    plt.close()

def analyze_attention_patterns():
    """Analyze what regions the model focuses on for different predictions"""
    import random
    
    print("\nAnalyzing attention patterns...")
    
    # Sample images from different countries
    countries = [c for c in os.listdir(DATA_DIR) 
                if os.path.isdir(os.path.join(DATA_DIR, c)) and not c.startswith('.')]
    
    sampled_countries = random.sample(countries, min(10, len(countries)))
    
    for country in sampled_countries:
        country_path = os.path.join(DATA_DIR, country)
        images = [f for f in os.listdir(country_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            img_path = os.path.join(country_path, images[0])
            save_path = os.path.join(OUTPUT_DIR, f"attention_{country.replace(' ', '_')}.png")
            
            result = visualize_single_prediction(img_path, save_path)
            
            pred_country, conf = result['country']
            status = "✓ CORRECT" if pred_country == country else "✗ INCORRECT"
            print(f"{country}: {pred_country} ({conf:.1f}%) {status}")

if __name__ == "__main__":
    print("="*60)
    print("ATTENTION VISUALIZATION")
    print("="*60)
    
    # Create grid visualization
    print("\n1. Creating attention grid...")
    visualize_multiple_samples(num_samples=6)
    
    # Analyze patterns
    print("\n2. Analyzing attention patterns for individual countries...")
    analyze_attention_patterns()
    
    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)
