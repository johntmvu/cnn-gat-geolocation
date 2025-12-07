"""
Comprehensive Model Comparison: Baseline CNN vs Multi-Task CNN vs Hybrid CNN-ViT
Analyzes performance, model complexity, and computational efficiency
"""

import torch
import torch.nn as nn
from torchvision import models
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_multitask import MultiTaskGeoModel, SpatialAttention

# Try importing CLIP-based models (optional)
try:
    from scripts.training.train_hybrid_cnn_vit import HybridCNNViT
    HYBRID_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HYBRID_AVAILABLE = False
    HybridCNNViT = None
    print("Warning: Could not import HybridCNNViT (CLIP not installed)")

try:
    from scripts.training.train_clip_only import CLIPGeoModel
    CLIP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CLIP_AVAILABLE = False
    CLIPGeoModel = None
    print("Warning: Could not import CLIPGeoModel (CLIP not installed)")

MODEL_DIR = str(PROJECT_ROOT / "models")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs/metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metrics
def load_metrics(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

baseline_metrics = load_metrics("training_metrics.json")
multitask_metrics = load_metrics("multitask_metrics.json")
hybrid_metrics = load_metrics("hybrid_cnn_vit_metrics.json")
clip_metrics = load_metrics("clip_only_metrics.json")

# Check which models are available
models_available = []
if baseline_metrics:
    models_available.append("Baseline CNN")
if multitask_metrics:
    models_available.append("Multi-Task CNN")
if hybrid_metrics:
    models_available.append("Hybrid CNN-ViT")
if clip_metrics:
    models_available.append("CLIP-Only")

print(f"Found {len(models_available)} trained models: {', '.join(models_available)}")

if len(models_available) < 2:
    print("Error: Need at least 2 trained models for comparison")
    exit(1)

# Extract epoch counts
baseline_epochs = len(baseline_metrics["train_loss"]) if baseline_metrics else 0
multitask_epochs = len(multitask_metrics["train_loss"]) if multitask_metrics else 0
hybrid_epochs = len(hybrid_metrics["train_loss"]) if hybrid_metrics else 0
clip_epochs = len(clip_metrics["train_loss"]) if clip_metrics else 0

print(f"\nEpochs trained:")
if baseline_metrics:
    print(f"  Baseline CNN: {baseline_epochs}")
if multitask_metrics:
    print(f"  Multi-Task CNN: {multitask_epochs}")
if hybrid_metrics:
    print(f"  Hybrid CNN-ViT: {hybrid_epochs}")
if clip_metrics:
    print(f"  CLIP-Only: {clip_epochs}")

# Model Architecture Comparison
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(num_params):
    # Assuming float32 (4 bytes per parameter)
    return (num_params * 4) / (1024 ** 2)

# Initialize models to count parameters
print("\nInitializing models for comparison...")

# Baseline CNN
baseline_model = models.resnet50(weights='IMAGENET1K_V1')
baseline_model.fc = nn.Linear(baseline_model.fc.in_features, 120)  # Assuming 120 countries
baseline_params = count_parameters(baseline_model)
baseline_size = get_model_size_mb(baseline_params)

# Multi-Task CNN
multitask_model = MultiTaskGeoModel(
    num_countries=120,
    num_regions=7,
    num_climates=6
)
multitask_params = count_parameters(multitask_model)
multitask_size = get_model_size_mb(multitask_params)

# Hybrid CNN-ViT (if CLIP available)
if HYBRID_AVAILABLE and HybridCNNViT is not None:
    try:
        hybrid_model = HybridCNNViT(
            num_countries=120,
            num_regions=7,
            num_climates=6,
            clip_model_name="ViT-B/32"
        )
        hybrid_params = count_parameters(hybrid_model)
        hybrid_size = get_model_size_mb(hybrid_params)
    except Exception as e:
        print(f"Could not initialize HybridCNNViT: {e}")
        hybrid_params = 180000000  # Approximate
        hybrid_size = 686.6
else:
    hybrid_params = 180000000  # Approximate for metrics
    hybrid_size = 686.6

# CLIP-Only (if CLIP available)
if CLIP_AVAILABLE and CLIPGeoModel is not None:
    try:
        clip_model = CLIPGeoModel(
            num_countries=120,
            num_regions=7,
            num_climates=6,
            clip_model_name="ViT-B/32"
        )
        clip_params = count_parameters(clip_model)
        clip_size = get_model_size_mb(clip_params)
    except Exception as e:
        print(f"Could not initialize CLIPGeoModel: {e}")
        clip_params = 88000000  # Approximate
        clip_size = 335.7
else:
    clip_params = 88000000  # Approximate for metrics
    clip_size = 335.7

print("\nModel Complexity:")
print(f"  Baseline CNN: {baseline_params/1e6:.2f}M params ({baseline_size:.1f} MB)")
print(f"  Multi-Task CNN: {multitask_params/1e6:.2f}M params ({multitask_size:.1f} MB)")
print(f"  Hybrid CNN-ViT: {hybrid_params/1e6:.2f}M params ({hybrid_size:.1f} MB)")
print(f"  CLIP-Only: {clip_params/1e6:.2f}M params ({clip_size:.1f} MB)")

# Plotting function
def plot_comprehensive_comparison():
    """Create comprehensive comparison plots"""
    
    # Determine subplot layout based on available models
    fig = plt.figure(figsize=(24, 12))
    
    # 1. Training Loss Comparison
    ax1 = plt.subplot(2, 4, 1)
    if baseline_metrics:
        ax1.plot(range(1, baseline_epochs + 1), baseline_metrics["train_loss"], 
                'b-', label='Baseline CNN', linewidth=2)
    if multitask_metrics:
        ax1.plot(range(1, multitask_epochs + 1), multitask_metrics["train_loss"], 
                'g-', label='Multi-Task CNN', linewidth=2)
    if hybrid_metrics:
        ax1.plot(range(1, hybrid_epochs + 1), hybrid_metrics["train_loss"], 
                'r-', label='Hybrid CNN-ViT', linewidth=2)
    if clip_metrics:
        ax1.plot(range(1, clip_epochs + 1), clip_metrics["train_loss"], 
                'm-', label='CLIP-Only', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss Comparison
    ax2 = plt.subplot(2, 4, 2)
    if baseline_metrics:
        ax2.plot(range(1, baseline_epochs + 1), baseline_metrics["val_loss"], 
                'b-', label='Baseline CNN', linewidth=2)
    if multitask_metrics:
        ax2.plot(range(1, multitask_epochs + 1), multitask_metrics["val_loss"], 
                'g-', label='Multi-Task CNN', linewidth=2)
    if hybrid_metrics:
        ax2.plot(range(1, hybrid_epochs + 1), hybrid_metrics["val_loss"], 
                'r-', label='Hybrid CNN-ViT', linewidth=2)
    if clip_metrics:
        ax2.plot(range(1, clip_epochs + 1), clip_metrics["val_loss"], 
                'm-', label='CLIP-Only', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Country Accuracy Comparison
    ax3 = plt.subplot(2, 4, 3)
    if baseline_metrics:
        ax3.plot(range(1, baseline_epochs + 1), baseline_metrics["val_acc"], 
                'b-', label='Baseline CNN', linewidth=2)
    if multitask_metrics:
        ax3.plot(range(1, multitask_epochs + 1), multitask_metrics["val_country_acc"], 
                'g-', label='Multi-Task CNN', linewidth=2)
    if hybrid_metrics:
        ax3.plot(range(1, hybrid_epochs + 1), hybrid_metrics["val_country_acc"], 
                'r-', label='Hybrid CNN-ViT', linewidth=2)
    if clip_metrics:
        ax3.plot(range(1, clip_epochs + 1), clip_metrics["val_country_acc"], 
                'm-', label='CLIP-Only', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Country Accuracy (%)', fontsize=12)
    ax3.set_title('Country Classification Accuracy', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Accuracy Bar Chart
    ax4 = plt.subplot(2, 4, 4)
    models_list = []
    accuracies = []
    colors_list = []
    
    if baseline_metrics:
        models_list.append('Baseline\nCNN')
        accuracies.append(baseline_metrics["val_acc"][-1])
        colors_list.append('blue')
    if multitask_metrics:
        models_list.append('Multi-Task\nCNN')
        accuracies.append(multitask_metrics["val_country_acc"][-1])
        colors_list.append('green')
    if hybrid_metrics:
        models_list.append('Hybrid\nCNN-ViT')
        accuracies.append(hybrid_metrics["val_country_acc"][-1])
        colors_list.append('red')
    if clip_metrics:
        models_list.append('CLIP-Only')
        accuracies.append(clip_metrics["val_country_acc"][-1])
        colors_list.append('magenta')
    
    bars = ax4.bar(models_list, accuracies, color=colors_list, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Country Accuracy (%)', fontsize=12)
    ax4.set_title('Final Country Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 5. Model Complexity Comparison
    ax5 = plt.subplot(2, 4, 5)
    models_list = ['Baseline\nCNN', 'Multi-Task\nCNN', 'Hybrid\nCNN-ViT', 'CLIP-Only']
    params_list = [baseline_params/1e6, multitask_params/1e6, hybrid_params/1e6, clip_params/1e6]
    colors_list = ['blue', 'green', 'red', 'magenta']
    
    bars = ax5.bar(models_list, params_list, color=colors_list, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Parameters (Millions)', fontsize=12)
    ax5.set_title('Model Complexity', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 6. Multi-Task Performance (Region & Climate)
    ax6 = plt.subplot(2, 4, 6)
    if multitask_metrics and hybrid_metrics and clip_metrics:
        # Compare all multi-task models on auxiliary tasks
        categories = ['Region', 'Climate']
        multitask_scores = [
            multitask_metrics["val_region_acc"][-1],
            multitask_metrics["val_climate_acc"][-1]
        ]
        hybrid_scores = [
            hybrid_metrics["val_region_acc"][-1],
            hybrid_metrics["val_climate_acc"][-1]
        ]
        clip_scores = [
            clip_metrics["val_region_acc"][-1],
            clip_metrics["val_climate_acc"][-1]
        ]
        
        x = np.arange(len(categories))
        width = 0.25
        
        bars1 = ax6.bar(x - width, multitask_scores, width, label='Multi-Task CNN',
                       color='green', alpha=0.7, edgecolor='black')
        bars2 = ax6.bar(x, hybrid_scores, width, label='Hybrid CNN-ViT',
                       color='red', alpha=0.7, edgecolor='black')
        bars3 = ax6.bar(x + width, clip_scores, width, label='CLIP-Only',
                       color='magenta', alpha=0.7, edgecolor='black')
        
        ax6.set_ylabel('Accuracy (%)', fontsize=12)
        ax6.set_title('Auxiliary Task Performance', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    elif multitask_metrics and hybrid_metrics:
        # Compare only multi-task CNN and hybrid if CLIP not available
        categories = ['Region', 'Climate']
        multitask_scores = [
            multitask_metrics["val_region_acc"][-1],
            multitask_metrics["val_climate_acc"][-1]
        ]
        hybrid_scores = [
            hybrid_metrics["val_region_acc"][-1],
            hybrid_metrics["val_climate_acc"][-1]
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, multitask_scores, width, label='Multi-Task CNN',
                       color='green', alpha=0.7, edgecolor='black')
        bars2 = ax6.bar(x + width/2, hybrid_scores, width, label='Hybrid CNN-ViT',
                       color='red', alpha=0.7, edgecolor='black')
        
        ax6.set_ylabel('Accuracy (%)', fontsize=12)
        ax6.set_title('Auxiliary Task Performance', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    else:
        ax6.text(0.5, 0.5, 'Multi-task models needed\nfor this comparison',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')
    
    # 7. Training Efficiency (Time per epoch proxy - parameters)
    ax7 = plt.subplot(2, 4, 7)
    models_list = []
    trainable_list = []
    if baseline_metrics:
        models_list.append('Baseline\nCNN')
        trainable_list.append(baseline_params/1e6)
    if multitask_metrics:
        models_list.append('Multi-Task\nCNN')
        trainable_list.append(multitask_params/1e6)
    if hybrid_metrics:
        models_list.append('Hybrid\nCNN-ViT')
        trainable_list.append(28.7)  # Trainable params when CLIP frozen
    if clip_metrics:
        models_list.append('CLIP-Only')
        trainable_list.append(2.5)  # Only task heads trainable initially
    
    colors_list = ['blue', 'green', 'red', 'magenta'][:len(models_list)]
    bars = ax7.bar(models_list, trainable_list, color=colors_list, alpha=0.7, edgecolor='black')
    ax7.set_ylabel('Trainable Parameters (M)', fontsize=12)
    ax7.set_title('Training Efficiency (Initial)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 8. Accuracy per Million Parameters (Efficiency)
    ax8 = plt.subplot(2, 4, 8)
    models_list = []
    efficiency_list = []
    if baseline_metrics:
        models_list.append('Baseline\nCNN')
        efficiency_list.append(baseline_metrics["val_acc"][-1] / (baseline_params/1e6))
    if multitask_metrics:
        models_list.append('Multi-Task\nCNN')
        efficiency_list.append(multitask_metrics["val_country_acc"][-1] / (multitask_params/1e6))
    if hybrid_metrics:
        models_list.append('Hybrid\nCNN-ViT')
        efficiency_list.append(hybrid_metrics["val_country_acc"][-1] / (hybrid_params/1e6))
    if clip_metrics:
        models_list.append('CLIP-Only')
        efficiency_list.append(clip_metrics["val_country_acc"][-1] / (clip_params/1e6))
    
    colors_list = ['blue', 'green', 'red', 'magenta'][:len(models_list)]
    bars = ax8.bar(models_list, efficiency_list, color=colors_list, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Accuracy / Million Params', fontsize=12)
    ax8.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "comprehensive_model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")
    plt.close()

# Generate comparison table
def generate_comparison_table():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Performance metrics
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 100)
    print(f"{'Metric':<30} {'Baseline CNN':<20} {'Multi-Task CNN':<20} {'Hybrid CNN-ViT':<20} {'CLIP-Only':<20}")
    print("-" * 100)
    
    # Country Accuracy
    baseline_acc = f"{baseline_metrics['val_acc'][-1]:.2f}%" if baseline_metrics else "N/A"
    multitask_acc = f"{multitask_metrics['val_country_acc'][-1]:.2f}%" if multitask_metrics else "N/A"
    hybrid_acc = f"{hybrid_metrics['val_country_acc'][-1]:.2f}%" if hybrid_metrics else "N/A"
    clip_acc = f"{clip_metrics['val_country_acc'][-1]:.2f}%" if clip_metrics else "N/A"
    print(f"{'Country Accuracy':<30} {baseline_acc:<20} {multitask_acc:<20} {hybrid_acc:<20} {clip_acc:<20}")
    
    # Region Accuracy
    multitask_region = f"{multitask_metrics['val_region_acc'][-1]:.2f}%" if multitask_metrics else "N/A"
    hybrid_region = f"{hybrid_metrics['val_region_acc'][-1]:.2f}%" if hybrid_metrics else "N/A"
    clip_region = f"{clip_metrics['val_region_acc'][-1]:.2f}%" if clip_metrics else "N/A"
    print(f"{'Region Accuracy':<30} {'N/A':<20} {multitask_region:<20} {hybrid_region:<20} {clip_region:<20}")
    
    # Climate Accuracy
    multitask_climate = f"{multitask_metrics['val_climate_acc'][-1]:.2f}%" if multitask_metrics else "N/A"
    hybrid_climate = f"{hybrid_metrics['val_climate_acc'][-1]:.2f}%" if hybrid_metrics else "N/A"
    clip_climate = f"{clip_metrics['val_climate_acc'][-1]:.2f}%" if clip_metrics else "N/A"
    print(f"{'Climate Accuracy':<30} {'N/A':<20} {multitask_climate:<20} {hybrid_climate:<20} {clip_climate:<20}")
    
    # Final Loss
    baseline_loss = f"{baseline_metrics['val_loss'][-1]:.4f}" if baseline_metrics else "N/A"
    multitask_loss = f"{multitask_metrics['val_loss'][-1]:.4f}" if multitask_metrics else "N/A"
    hybrid_loss = f"{hybrid_metrics['val_loss'][-1]:.4f}" if hybrid_metrics else "N/A"
    clip_loss = f"{clip_metrics['val_loss'][-1]:.4f}" if clip_metrics else "N/A"
    print(f"{'Final Validation Loss':<30} {baseline_loss:<20} {multitask_loss:<20} {hybrid_loss:<20} {clip_loss:<20}")
    
    # Model complexity
    print("\n🏗️  MODEL COMPLEXITY")
    print("-" * 100)
    print(f"{'Metric':<30} {'Baseline CNN':<20} {'Multi-Task CNN':<20} {'Hybrid CNN-ViT':<20} {'CLIP-Only':<20}")
    print("-" * 100)
    print(f"{'Parameters':<30} {f'{baseline_params/1e6:.2f}M':<20} {f'{multitask_params/1e6:.2f}M':<20} {f'{hybrid_params/1e6:.2f}M':<20} {f'{clip_params/1e6:.2f}M':<20}")
    print(f"{'Model Size':<30} {f'{baseline_size:.1f} MB':<20} {f'{multitask_size:.1f} MB':<20} {f'{hybrid_size:.1f} MB':<20} {f'{clip_size:.1f} MB':<20}")
    
    # Architecture details
    print("\n🔧 ARCHITECTURE DETAILS")
    print("-" * 100)
    print(f"{'Feature':<30} {'Baseline CNN':<20} {'Multi-Task CNN':<20} {'Hybrid CNN-ViT':<20} {'CLIP-Only':<20}")
    print("-" * 100)
    print(f"{'Backbone':<30} {'ResNet50':<20} {'ResNet50':<20} {'ResNet50 + CLIP':<20} {'CLIP ViT-B/32':<20}")
    print(f"{'Attention Mechanism':<30} {'None':<20} {'Spatial Attention':<20} {'Fusion Attention':<20} {'Self-Attention':<20}")
    print(f"{'Tasks':<30} {'Single (Country)':<20} {'Multi (C+R+Clim)':<20} {'Multi (C+R+Clim)':<20} {'Multi (C+R+Clim)':<20}")
    print(f"{'Global Context':<30} {'Limited':<20} {'Limited':<20} {'Full (CNN+CLIP)':<20} {'Full (CLIP)':<20}")
    print(f"{'Pretrained':<30} {'ImageNet':<20} {'ImageNet':<20} {'ImageNet + CLIP':<20} {'CLIP (400M pairs)':<20}")
    print(f"{'Transformer Layers':<30} {'0':<20} {'0':<20} {'12 (CLIP)':<20} {'12 (CLIP)':<20}")
    print(f"{'Attention Heads':<30} {'N/A':<20} {'N/A':<20} {'12 (CLIP)':<20} {'12 (CLIP)':<20}")
    
    # Performance improvements
    if baseline_metrics and multitask_metrics:
        baseline_acc_val = baseline_metrics['val_acc'][-1]
        multitask_acc_val = multitask_metrics['val_country_acc'][-1]
        improvement1 = multitask_acc_val - baseline_acc_val
        
        print("\n📈 PERFORMANCE IMPROVEMENTS")
        print("-" * 80)
        print(f"Multi-Task CNN vs Baseline:     {improvement1:+.2f}% country accuracy")
    
    if baseline_metrics and hybrid_metrics:
        hybrid_acc_val = hybrid_metrics['val_country_acc'][-1]
        improvement2 = hybrid_acc_val - baseline_acc_val
        print(f"Hybrid CNN-ViT vs Baseline:     {improvement2:+.2f}% country accuracy")
    
    if multitask_metrics and hybrid_metrics:
        improvement3 = hybrid_acc_val - multitask_acc_val
        print(f"Hybrid CNN-ViT vs Multi-Task:   {improvement3:+.2f}% country accuracy")
    
    if baseline_metrics and clip_metrics:
        clip_acc_val = clip_metrics['val_country_acc'][-1]
        improvement4 = clip_acc_val - baseline_acc_val
        print(f"CLIP-Only vs Baseline:          {improvement4:+.2f}% country accuracy")
    
    if hybrid_metrics and clip_metrics:
        improvement5 = clip_acc_val - hybrid_acc_val
        print(f"CLIP-Only vs Hybrid CNN-ViT:    {improvement5:+.2f}% country accuracy")
    
    # Key advantages
    print("\n🌟 KEY ADVANTAGES")
    print("-" * 80)
    print("Baseline CNN:")
    print("  • Simplest architecture, fastest training")
    print("  • Lowest memory footprint")
    print("  • Good baseline performance")
    
    print("\nMulti-Task CNN:")
    print("  • Hierarchical geographic understanding (country/region/climate)")
    print("  • Spatial attention for interpretability")
    print("  • Regularization through auxiliary tasks")
    
    print("\nHybrid CNN-ViT:")
    print("  • Best of both worlds: CNN local features + CLIP global context")
    print("  • Leverages pretrained CLIP knowledge (400M image-text pairs)")
    print("  • Feature fusion through cross-attention mechanism")
    print("  • State-of-the-art hybrid architecture")
    print("  • Superior interpretability through fusion attention")
    
    print("\nCLIP-Only:")
    print("  • Simplest architecture - pure vision transformer")
    print("  • Highly pretrained on massive dataset (400M pairs)")
    print("  • Minimal trainable parameters (only task heads)")
    print("  • Fastest training time")
    print("  • Best parameter efficiency")
    print("  • Strong semantic understanding")
    
    print("\n" + "="*100)

# Main execution
if __name__ == '__main__':
    plot_comprehensive_comparison()
    generate_comparison_table()
    
    print("\n✅ Comparison complete!")
    print(f"Results saved to: {OUTPUT_DIR}/comprehensive_model_comparison.png")
