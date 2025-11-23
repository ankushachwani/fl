"""
Visualization utilities for model inversion attacks
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def visualize_attack_results(
    original_images: Optional[torch.Tensor],
    reconstructed_images: torch.Tensor,
    labels: Optional[torch.Tensor],
    attack_name: str,
    save_path: str = 'attack_results.png',
    denormalize: bool = True
):
    """
    Visualize original and reconstructed images side by side.
    
    Args:
        original_images: Original training images (optional)
        reconstructed_images: Images reconstructed by attack
        labels: Class labels
        attack_name: Name of the attack for the title
        save_path: Path to save the figure
        denormalize: Whether to denormalize images
    """
    num_images = reconstructed_images.shape[0]
    has_originals = original_images is not None
    
    # Setup figure
    if has_originals:
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1) if has_originals else [axes]
    
    # Convert to numpy and denormalize if needed
    def to_numpy(img):
        img = img.detach().cpu()
        if denormalize:
            # Denormalize from [-1, 1] or normalized range
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.squeeze().numpy()
    
    for i in range(num_images):
        # Plot original if available
        if has_originals:
            orig_img = to_numpy(original_images[i])
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].set_title(f'Original\nLabel: {labels[i].item() if labels is not None else "?"}')
            axes[0, i].axis('off')
            
            # Plot reconstructed
            recon_img = to_numpy(reconstructed_images[i])
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        else:
            # Only reconstructed images
            recon_img = to_numpy(reconstructed_images[i])
            if num_images == 1:
                axes[0].imshow(recon_img, cmap='gray')
                axes[0].set_title(f'{attack_name}\nClass: {labels[i].item() if labels is not None else i}')
                axes[0].axis('off')
            else:
                axes[i].imshow(recon_img, cmap='gray')
                axes[i].set_title(f'Class: {labels[i].item() if labels is not None else i}')
                axes[i].axis('off')
    
    plt.suptitle(f'{attack_name} Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def visualize_all_classes(
    reconstructed_images: torch.Tensor,
    num_classes: int,
    attack_name: str,
    save_path: str = 'all_classes.png'
):
    """
    Visualize reconstructed images for all classes in a grid.
    
    Args:
        reconstructed_images: Tensor of shape (num_classes, C, H, W)
        num_classes: Number of classes
        attack_name: Name of the attack
        save_path: Path to save the figure
    """
    cols = min(5, num_classes)
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Handle different subplot configurations
    if num_classes == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = np.array(axes).reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(num_classes):
        img = reconstructed_images[i].detach().cpu().squeeze().numpy()
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Class {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{attack_name} - All Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved all classes visualization to {save_path}")
    plt.close()


def plot_attack_loss_history(
    history: List,
    attack_name: str,
    save_path: str = 'loss_history.png'
):
    """
    Plot the loss history during attack optimization.
    
    Args:
        history: List of loss values or dictionaries
        attack_name: Name of the attack
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(history[0], dict):
        # Multiple loss components
        keys = history[0].keys()
        for key in keys:
            values = [h[key] for h in history]
            plt.plot(values, label=key, linewidth=2)
        plt.legend()
    else:
        # Single loss value
        plt.plot(history, linewidth=2, color='blue')
        plt.ylabel('Loss')
    
    plt.xlabel('Iteration')
    plt.title(f'{attack_name} - Optimization History')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss history to {save_path}")
    plt.close()


def compare_attacks(
    results_dict: dict,
    save_path: str = 'attack_comparison.png'
):
    """
    Compare reconstructions from multiple attacks side by side.
    
    Args:
        results_dict: Dictionary mapping attack names to reconstructed images
        save_path: Path to save the figure
    """
    num_attacks = len(results_dict)
    attack_names = list(results_dict.keys())
    
    # Assume all have same number of images
    num_images = results_dict[attack_names[0]].shape[0]
    
    fig, axes = plt.subplots(num_attacks, num_images, 
                            figsize=(num_images * 2, num_attacks * 2))
    
    if num_attacks == 1:
        axes = axes.reshape(1, -1)
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i, attack_name in enumerate(attack_names):
        images = results_dict[attack_name]
        
        for j in range(num_images):
            img = images[j].detach().cpu().squeeze().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[i, j].imshow(img, cmap='gray')
            if j == 0:
                axes[i, j].set_ylabel(attack_name, fontsize=12, fontweight='bold')
            if i == 0:
                axes[i, j].set_title(f'Class {j}')
            axes[i, j].axis('off')
    
    plt.suptitle('Attack Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved attack comparison to {save_path}")
    plt.close()
