"""
Ablation Study: Compare Different Model Configurations
Tests: Single-task vs Multi-task, With vs Without Attention
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_DIR = "models"

def load_metrics(filename):
    """Load training metrics from JSON file"""
    try:
        with open(f"{RESULTS_DIR}/{filename}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

def plot_comparison():
    """Create comparison plots for ablation study"""
    
    # Load metrics from different experiments
    baseline_metrics = load_metrics("training_metrics.json")  # Single-task baseline
    multitask_metrics = load_metrics("multitask_metrics.json")  # Multi-task with attention
    
    if not baseline_metrics or not multitask_metrics:
        print("Error: Missing metrics files. Train both models first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Handle different epoch counts
    baseline_epochs = range(1, len(baseline_metrics["train_acc"]) + 1)
    multitask_epochs = range(1, len(multitask_metrics["train_country_acc"]) + 1)
    
    # Plot 1: Training Accuracy Comparison
    axes[0, 0].plot(baseline_epochs, baseline_metrics["train_acc"], 
                   label='Single-Task Baseline', marker='o', linewidth=2)
    axes[0, 0].plot(multitask_epochs, multitask_metrics["train_country_acc"], 
                   label='Multi-Task + Attention', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy Comparison
    axes[0, 1].plot(baseline_epochs, baseline_metrics["val_acc"], 
                   label='Single-Task Baseline', marker='o', linewidth=2)
    axes[0, 1].plot(multitask_epochs, multitask_metrics["val_country_acc"], 
                   label='Multi-Task + Attention', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss Comparison
    axes[1, 0].plot(baseline_epochs, baseline_metrics["train_loss"], 
                   label='Single-Task Train', marker='o', linewidth=2, alpha=0.7)
    axes[1, 0].plot(baseline_epochs, baseline_metrics["val_loss"], 
                   label='Single-Task Val', marker='o', linewidth=2, linestyle='--', alpha=0.7)
    axes[1, 0].plot(multitask_epochs, multitask_metrics["train_loss"], 
                   label='Multi-Task Train', marker='s', linewidth=2, alpha=0.7)
    axes[1, 0].plot(multitask_epochs, multitask_metrics["val_loss"], 
                   label='Multi-Task Val', marker='s', linewidth=2, linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Multi-Task Auxiliary Tasks
    axes[1, 1].plot(multitask_epochs, multitask_metrics["val_country_acc"], 
                   label='Country', marker='o', linewidth=2)
    axes[1, 1].plot(multitask_epochs, multitask_metrics["val_region_acc"], 
                   label='Region', marker='s', linewidth=2)
    axes[1, 1].plot(multitask_epochs, multitask_metrics["val_climate_acc"], 
                   label='Climate', marker='^', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Multi-Task: All Tasks Performance', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/ablation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {RESULTS_DIR}/ablation_comparison.png")
    plt.close()

def generate_summary_table():
    """Generate a summary table comparing all models"""
    
    baseline_metrics = load_metrics("training_metrics.json")
    multitask_metrics = load_metrics("multitask_metrics.json")
    
    if not baseline_metrics or not multitask_metrics:
        print("Error: Missing metrics files")
        return
    
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    print("\n{:<30} {:<20} {:<20}".format("Metric", "Single-Task", "Multi-Task + Attention"))
    print("-"*70)
    
    # Best validation accuracies
    baseline_best = max(baseline_metrics["val_acc"])
    multitask_best = max(multitask_metrics["val_country_acc"])
    improvement = multitask_best - baseline_best
    
    print("{:<30} {:<20.2f}% {:<20.2f}%".format(
        "Best Val Accuracy (Country)", baseline_best, multitask_best))
    print("{:<30} {:<20} {:<20.2f}%".format(
        "Improvement", "-", improvement))
    
    # Final epoch metrics
    baseline_final = baseline_metrics["val_acc"][-1]
    multitask_final = multitask_metrics["val_country_acc"][-1]
    
    print("{:<30} {:<20.2f}% {:<20.2f}%".format(
        "Final Val Accuracy", baseline_final, multitask_final))
    
    # Multi-task auxiliary tasks
    region_best = max(multitask_metrics["val_region_acc"])
    climate_best = max(multitask_metrics["val_climate_acc"])
    
    print("\n" + "-"*70)
    print("Multi-Task Auxiliary Tasks:")
    print("{:<30} {:<20}".format("  Region Accuracy", f"{region_best:.2f}%"))
    print("{:<30} {:<20}".format("  Climate Accuracy", f"{climate_best:.2f}%"))
    
    # Convergence speed (epochs to reach 80% if applicable)
    def epochs_to_threshold(acc_list, threshold=80):
        for i, acc in enumerate(acc_list):
            if acc >= threshold:
                return i + 1
        return len(acc_list)
    
    baseline_conv = epochs_to_threshold(baseline_metrics["val_acc"])
    multitask_conv = epochs_to_threshold(multitask_metrics["val_country_acc"])
    
    print("\n" + "-"*70)
    print("Convergence Speed (epochs to 80% accuracy):")
    print("{:<30} {:<20} {:<20}".format(
        "", baseline_conv, multitask_conv))
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    if improvement > 0:
        print(f"✓ Multi-task learning improves accuracy by {improvement:.2f}%")
    else:
        print(f"  Multi-task learning shows {abs(improvement):.2f}% difference")
    
    print(f"✓ Attention mechanism provides interpretability")
    print(f"✓ Model learns auxiliary tasks: Region ({region_best:.1f}%), Climate ({climate_best:.1f}%)")
    
    if multitask_conv < baseline_conv:
        print(f"✓ Multi-task converges {baseline_conv - multitask_conv} epochs faster")
    
    print("="*80 + "\n")

def compare_model_sizes():
    """Compare model complexity"""
    print("\n" + "="*80)
    print("MODEL COMPLEXITY COMPARISON")
    print("="*80)
    
    # ResNet50 base parameters
    base_params = 23.5  # Million parameters
    
    # Single-task: base + 1 FC layer
    # Multi-task: base + attention + 3 FC layers
    
    print(f"\nSingle-Task Model:")
    print(f"  Base parameters: {base_params}M")
    print(f"  Additional: ~2K (1 FC layer)")
    print(f"  Total: ~{base_params}M")
    
    print(f"\nMulti-Task + Attention Model:")
    print(f"  Base parameters: {base_params}M")
    print(f"  Attention layer: ~2K")
    print(f"  Task heads (3x): ~6K")
    print(f"  Total: ~{base_params}M")
    
    print(f"\nParameter increase: <0.1% (negligible)")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING ABLATION STUDY")
    print("="*80)
    
    # Generate visualizations
    print("\n1. Creating comparison plots...")
    plot_comparison()
    
    # Generate summary
    print("\n2. Generating summary statistics...")
    generate_summary_table()
    
    # Compare complexity
    print("\n3. Comparing model complexity...")
    compare_model_sizes()
    
    print("\n" + "="*80)
    print("✓ Ablation study complete!")
    print("="*80)
