"""
evaluate_model.py — Evaluate trained ResNet50 model on validation/test split
Generates sample predictions, accuracy metrics, and Grad-CAM visualizations.
"""

import os
import random
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from train_model import LandmarkDataset, val_ds, label2id, id2label, DEVICE, MODEL_DIR
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================

model_path = os.path.join(MODEL_DIR, "best_resnet50.pth")
num_classes = len(label2id)

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval().to(DEVICE)

print(f"Loaded model from {model_path}")

# Load the mapping CSV
mapping_df = pd.read_csv("data/gldv2_micro/train_label_to_category.csv")
id_to_location = dict(zip(mapping_df["landmark_id"], mapping_df["category"]))

# Function to map landmark_id to location category
def map_landmark_to_location(landmark_id):
    return id_to_location.get(landmark_id, "Unknown")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

indices = random.sample(range(len(val_ds)), 8)
tfm_reverse = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

plt.figure(figsize=(16, 8))
for i, idx in enumerate(indices):
    img, label = val_ds[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(DEVICE))
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    location = map_landmark_to_location(pred)
    print(f"Predicted landmark_id: {pred}, Location: {location}")
    unnorm = tfm_reverse(img).permute(1, 2, 0).clamp(0, 1)
    plt.subplot(2, 4, i + 1)
    plt.imshow(unnorm)
    plt.title(f"True: {id2label[label]}\nPred: {id2label[pred]}")
    plt.axis("off")

plt.tight_layout()
os.makedirs("outputs/predictions", exist_ok=True)
plt.savefig("outputs/predictions/sample_predictions.png")
plt.show()

# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

img, label = val_ds[random.randint(0, len(val_ds))]
rgb_img = img.permute(1,2,0).cpu().numpy()
input_tensor = img.unsqueeze(0).to(DEVICE)
grayscale_cam = cam(input_tensor=input_tensor)[0, :]
vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

os.makedirs("outputs/gradcam", exist_ok=True)
plt.imshow(vis)
plt.title(f"Grad-CAM | True: {id2label[label]}")
plt.axis("off")
plt.savefig("outputs/gradcam/example_cam.png")
plt.show()

print("Evaluation complete.")
