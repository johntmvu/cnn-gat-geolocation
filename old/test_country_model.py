"""
Test Country Classification Model
Load a trained model and predict country from an image
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuration
MODEL_PATH = "models/best_country_model.pth"
MAPPING_PATH = "models/country_mapping.json"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_country(image_path):
    """Predict country from an image"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_country = idx_to_country[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    
    return predicted_country, confidence_pct

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_country_model.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    country, confidence = predict_country(image_path)
    print(f"Predicted Country: {country}")
    print(f"Confidence: {confidence:.2f}%")
