"""
Test Country Model with Random Images
Evaluates the trained model on random samples from the dataset
Shows predictions with confidence scores and actual labels
"""

import os
import random
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import hybrid model
try:
    from scripts.training.train_hybrid_cnn_vit import HybridCNNViT
    MODEL_NAME = "hybrid_cnn_vit"
    MODEL_PATH = str(PROJECT_ROOT / "models/best_hybrid_cnn_vit.pth")
    MAPPINGS_PATH = str(PROJECT_ROOT / "models/hybrid_cnn_vit_mappings.json")
    USE_HYBRID = True
except ImportError:
    print("Warning: CLIP not available, falling back to baseline model")
    MODEL_NAME = "baseline_cnn"
    MODEL_PATH = str(PROJECT_ROOT / "models/best_country_model.pth")
    MAPPINGS_PATH = str(PROJECT_ROOT / "models/country_mapping.json")
    USE_HYBRID = False

DATA_DIR = str(PROJECT_ROOT / "data/gsv_50k/compressed_dataset")
OUTPUT_DIR = str(PROJECT_ROOT / f"outputs/predictions/{MODEL_NAME}")
IMG_SIZE = 224
NUM_SAMPLES = 10  # Number of random images to test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")

# Load mappings
with open(MAPPINGS_PATH, "r") as f:
    if USE_HYBRID:
        mappings = json.load(f)
        country_to_idx = mappings["country_to_idx"]
        region_to_idx = mappings.get("region_to_idx", {})
        climate_to_idx = mappings.get("climate_to_idx", {})
    else:
        country_to_idx = json.load(f)
        region_to_idx = {}
        climate_to_idx = {}

idx_to_country = {v: k for k, v in country_to_idx.items()}
idx_to_region = {v: k for k, v in region_to_idx.items()} if region_to_idx else {}
idx_to_climate = {v: k for k, v in climate_to_idx.items()} if climate_to_idx else {}

# Load model
num_countries = len(country_to_idx)
num_regions = len(region_to_idx) if region_to_idx else 7
num_climates = len(climate_to_idx) if climate_to_idx else 6

if USE_HYBRID:
    model = HybridCNNViT(
        num_countries=num_countries,
        num_regions=num_regions,
        num_climates=num_climates,
        clip_model_name="ViT-B/32"
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded Hybrid CNN-ViT: {num_countries} countries, {num_regions} regions, {num_climates} climates")
else:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_countries)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded Baseline CNN: {num_countries} countries")

model = model.to(DEVICE)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_random_images(data_dir, num_samples):
    """Get random images from the dataset with their true labels"""
    countries = [c for c in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, c)) and not c.startswith('.')]
    
    samples = []
    for _ in range(num_samples):
        country = random.choice(countries)
        country_path = os.path.join(data_dir, country)
        images = [f for f in os.listdir(country_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            img_name = random.choice(images)
            img_path = os.path.join(country_path, img_name)
            samples.append((img_path, country))
    
    return samples

def predict_image(image_path):
    """Predict country (and region/climate if hybrid) from an image"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        if USE_HYBRID:
            country_out, region_out, climate_out, _ = model(image_tensor)
            outputs = country_out
        else:
            outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
        
        predictions = []
        for prob, idx in zip(top3_prob[0], top3_idx[0]):
            country = idx_to_country[idx.item()]
            confidence = prob.item() * 100
            predictions.append((country, confidence))
        
        # Get region and climate predictions if using hybrid
        region_pred = None
        climate_pred = None
        if USE_HYBRID and region_to_idx and climate_to_idx:
            region_probs = torch.nn.functional.softmax(region_out, dim=1)
            climate_probs = torch.nn.functional.softmax(climate_out, dim=1)
            
            region_conf, region_idx = torch.max(region_probs, 1)
            climate_conf, climate_idx = torch.max(climate_probs, 1)
            
            region_pred = (idx_to_region[region_idx.item()], region_conf.item() * 100)
            climate_pred = (idx_to_climate[climate_idx.item()], climate_conf.item() * 100)
    
    return predictions, image, region_pred, climate_pred

def test_model():
    """Test model on random images and display results"""
    print(f"\nTesting model on {NUM_SAMPLES} random images...\n")
    print("="*80)
    
    samples = get_random_images(DATA_DIR, NUM_SAMPLES)
    
    correct = 0
    top3_correct = 0
    
    for idx, (img_path, true_country) in enumerate(samples, 1):
        predictions, image, region_pred, climate_pred = predict_image(img_path)
        predicted_country, confidence = predictions[0]
        
        is_correct = predicted_country == true_country
        is_top3 = true_country in [p[0] for p in predictions]
        
        if is_correct:
            correct += 1
        if is_top3:
            top3_correct += 1
        
        # Print results
        print(f"\nImage {idx}: {os.path.basename(img_path)}")
        print(f"True Country: {true_country}")
        print(f"Predicted: {predicted_country} (Confidence: {confidence:.2f}%)")
        
        if is_correct:
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
        
        print("\nTop 3 Predictions:")
        for rank, (country, conf) in enumerate(predictions, 1):
            marker = "✓" if country == true_country else " "
            print(f"  {rank}. {country}: {conf:.2f}% {marker}")
        
        # Print region and climate if available
        if region_pred and climate_pred:
            print(f"\nRegion: {region_pred[0]} (Confidence: {region_pred[1]:.2f}%)")
            print(f"Climate: {climate_pred[0]} (Confidence: {climate_pred[1]:.2f}%)")
        
        print("-"*80)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples: {NUM_SAMPLES}")
    print(f"Top-1 Accuracy: {correct}/{NUM_SAMPLES} ({100*correct/NUM_SAMPLES:.1f}%)")
    print(f"Top-3 Accuracy: {top3_correct}/{NUM_SAMPLES} ({100*top3_correct/NUM_SAMPLES:.1f}%)")
    print("="*80)

def visualize_predictions(num_display=6):
    """Visualize predictions with images"""
    samples = get_random_images(DATA_DIR, num_display)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (img_path, true_country) in enumerate(samples):
        predictions, image, region_pred, climate_pred = predict_image(img_path)
        predicted_country, confidence = predictions[0]
        
        # Display image
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        # Title with prediction
        color = 'green' if predicted_country == true_country else 'red'
        title = f"True: {true_country}\nPred: {predicted_country}\n({confidence:.1f}%)"
        
        # Add region and climate if available
        if region_pred and climate_pred:
            title += f"\nRegion: {region_pred[0]}\nClimate: {climate_pred[0]}"
        
        axes[idx].set_title(title, fontsize=9, color=color)
    
    plt.suptitle(f"Model: {MODEL_NAME.upper().replace('_', ' ')}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'test_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Test model with text output
    test_model()
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_predictions()
    
    print("\n✓ Testing complete!")
