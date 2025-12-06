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
MODEL_PATH = "models/best_country_model.pth"
MAPPING_PATH = "models/country_mapping.json"
DATA_DIR = "data/gsv_50k/compressed_dataset"
IMG_SIZE = 224
NUM_SAMPLES = 10  # Number of random images to test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Load country mapping
with open(MAPPING_PATH, "r") as f:
    country_to_idx = json.load(f)
idx_to_country = {v: k for k, v in country_to_idx.items()}

# Load model
num_classes = len(country_to_idx)
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"Model loaded with {num_classes} countries")

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
    """Predict country from an image"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
        
        predictions = []
        for prob, idx in zip(top3_prob[0], top3_idx[0]):
            country = idx_to_country[idx.item()]
            confidence = prob.item() * 100
            predictions.append((country, confidence))
    
    return predictions, image

def test_model():
    """Test model on random images and display results"""
    print(f"\nTesting model on {NUM_SAMPLES} random images...\n")
    print("="*80)
    
    samples = get_random_images(DATA_DIR, NUM_SAMPLES)
    
    correct = 0
    top3_correct = 0
    
    for idx, (img_path, true_country) in enumerate(samples, 1):
        predictions, image = predict_image(img_path)
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
        predictions, image = predict_image(img_path)
        predicted_country, confidence = predictions[0]
        
        # Display image
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        # Title with prediction
        color = 'green' if predicted_country == true_country else 'red'
        title = f"True: {true_country}\nPred: {predicted_country}\n({confidence:.1f}%)"
        axes[idx].set_title(title, fontsize=10, color=color)
    
    plt.tight_layout()
    plt.savefig('models/test_predictions.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: models/test_predictions.png")
    plt.close()

if __name__ == "__main__":
    # Test model with text output
    test_model()
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_predictions()
    
    print("\n✓ Testing complete!")
