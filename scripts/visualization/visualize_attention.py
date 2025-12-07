"""
Attention Visualization for Multi-Task Geolocation Model
Generates heatmaps showing which image regions the model focuses on
Uses Grad-CAM for hybrid models and spatial attention for multi-task models
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Try to import hybrid model first, fallback to multitask
try:
    from scripts.training.train_hybrid_cnn_vit import HybridCNNViT
    MODEL_NAME = "hybrid_cnn_vit"
    MODEL_PATH = str(PROJECT_ROOT / "models/best_hybrid_cnn_vit.pth")
    MAPPINGS_PATH = str(PROJECT_ROOT / "models/hybrid_cnn_vit_mappings.json")
    USE_HYBRID = True
    USE_GRADCAM = True
    print("Using Hybrid CNN-ViT model with Grad-CAM")
except ImportError:
    from scripts.training.train_multitask import MultiTaskGeoModel
    MODEL_NAME = "multitask_cnn"
    MODEL_PATH = str(PROJECT_ROOT / "models/best_multitask_model.pth")
    MAPPINGS_PATH = str(PROJECT_ROOT / "models/multitask_mappings.json")
    USE_HYBRID = False
    USE_GRADCAM = False
    print("CLIP not available, using Multi-Task CNN model with spatial attention")

# Configuration
DATA_DIR = str(PROJECT_ROOT / "data/gsv_50k/compressed_dataset")
OUTPUT_DIR = str(PROJECT_ROOT / f"outputs/attention_maps/{MODEL_NAME}")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")

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

if USE_HYBRID:
    model = HybridCNNViT(
        num_countries=num_countries,
        num_regions=num_regions,
        num_climates=num_climates,
        clip_model_name="ViT-B/32"
    )
else:
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

# Grad-CAM implementation
class GradCAM:
    """Grad-CAM for visualizing CNN attention"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        model_output = self.model(input_tensor)
        
        if isinstance(model_output, tuple):
            country_out = model_output[0]
        else:
            country_out = model_output
        
        # Backward pass for target class
        if target_class is None:
            target_class = country_out.argmax(dim=1)
        
        self.model.zero_grad()
        country_out[0, target_class].backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def predict_with_attention(image_path):
    """Predict and return attention map"""
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    if USE_GRADCAM:
        # Use Grad-CAM for hybrid model
        model.eval()
        
        # Get target layer (last conv layer of ResNet backbone)
        target_layer = model.cnn_backbone[-1][-1].conv3  # Last conv layer in ResNet50
        
        # Create Grad-CAM
        grad_cam = GradCAM(model, target_layer)
        
        # Generate attention map with gradients
        image_tensor.requires_grad = True
        attention = grad_cam.generate_cam(image_tensor)
        
        # Get predictions
        with torch.no_grad():
            country_out, region_out, climate_out, _ = model(image_tensor)
            
            country_probs = torch.nn.functional.softmax(country_out, dim=1)
            region_probs = torch.nn.functional.softmax(region_out, dim=1)
            climate_probs = torch.nn.functional.softmax(climate_out, dim=1)
            
            country_conf, country_idx = torch.max(country_probs, 1)
            region_conf, region_idx = torch.max(region_probs, 1)
            climate_conf, climate_idx = torch.max(climate_probs, 1)
            
            country = idx_to_country[country_idx.item()]
            region = idx_to_region[region_idx.item()]
            climate = idx_to_climate[climate_idx.item()]
    else:
        # Use spatial attention for multi-task model
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
            attention = attention_map[0, 0].cpu().numpy()
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

def visualize_single_prediction(image_path, save_path=None, true_country=None):
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
    
    # Include actual country if provided
    if true_country:
        status = "✓" if country == true_country else "✗"
        title = f"Model: {MODEL_NAME.upper().replace('_', ' ')}\n"
        title += f"Actual: {true_country} | Predicted: {country} ({country_conf:.1f}%) {status}\n"
    else:
        title = f"Model: {MODEL_NAME.upper().replace('_', ' ')}\n"
        title += f"Predicted Country: {country} ({country_conf:.1f}%)\n"
    
    title += f"Region: {region} ({region_conf:.1f}%) | "
    title += f"Climate: {climate} ({climate_conf:.1f}%)"
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
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
        status = "✓" if pred_country == true_country else "✗"
        pred_text = f"Actual: {true_country}\n"
        pred_text += f"Predicted: {pred_country} {status}\n({conf:.1f}%)\n\n"
        pred_text += f"Region:\n{result['region'][0]}\n({result['region'][1]:.1f}%)\n\n"
        pred_text += f"Climate:\n{result['climate'][0]}\n({result['climate'][1]:.1f}%)"
        
        color = 'lightgreen' if pred_country == true_country else 'lightcoral'
        axes[idx, 2].text(0.1, 0.5, pred_text, fontsize=10, 
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        if idx == 0:
            axes[idx, 2].set_title('Predictions', fontsize=12, fontweight='bold')
    
    plt.suptitle(f"Model: {MODEL_NAME.upper().replace('_', ' ')}", fontsize=16, fontweight='bold', y=0.9985)
    plt.tight_layout(rect=[0, 0, 1, 0.997])
    save_path = os.path.join(OUTPUT_DIR, "attention_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid visualization to: {save_path}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
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
            
            result = visualize_single_prediction(img_path, save_path, true_country=country)
            
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
