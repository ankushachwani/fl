"""
Complete attack suite with all three working attacks:
1. Gradient Inversion (iDLG)
2. Model Inversion (Confidence Maximization)
3. GAN-based Inversion (GMI)
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SimpleCNN
from src.attacks.gan_inversion import GANInversionAttack
from src.attacks.visualization import visualize_all_classes


def load_model(model_path: str, device: str):
    """Load trained model."""
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def prepare_data():
    """Prepare MNIST data."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Batch size 1 for gradient inversion
    grad_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )
    
    # Batch size 64 for GAN training
    gan_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True
    )
    
    return grad_loader, gan_loader


def gradient_inversion_attack(model, data_loader, device, num_samples=10):
    """
    Gradient Inversion using iDLG (Improved Deep Leakage from Gradients).
    Analytically recovers labels and reconstructs images from gradients.
    """
    print("\n" + "#"*70)
    print("# ATTACK 1: GRADIENT INVERSION (iDLG)")
    print("#"*70)
    
    all_reconstructed = []
    all_labels = []
    
    for sample_idx in range(num_samples):
        print(f"\n--- Sample {sample_idx + 1}/{num_samples} ---")
        
        # Get one sample
        data, labels = next(iter(data_loader))
        data, labels = data.to(device), labels.to(device)
        
        print(f"True label: {labels.item()}")
        
        # Compute true gradients
        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        true_gradients = torch.autograd.grad(loss, model.parameters())
        true_gradients = [g.detach().clone() for g in true_gradients]
        
        # Analytical label recovery (iDLG)
        last_grad = None
        for grad in reversed(true_gradients):
            if grad.dim() == 1:  # Bias gradient
                last_grad = grad
                break
        if last_grad is None:
            last_grad = true_gradients[-1].view(-1)
        recovered_label = torch.argmin(last_grad).item()
        print(f"Recovered label: {recovered_label}")
        
        # Initialize dummy data
        dummy_data = torch.randn_like(data, requires_grad=True)
        
        # Use Adam optimizer - much faster than LBFGS
        optimizer = torch.optim.Adam([dummy_data], lr=0.1)
        
        label_tensor = torch.tensor([recovered_label], device=device)
        
        # Optimize for fewer iterations with Adam
        for iteration in range(500):
            optimizer.zero_grad()
            
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, label_tensor)
            dummy_grads = torch.autograd.grad(
                dummy_loss, model.parameters(), create_graph=True
            )
            
            grad_diff = sum(
                ((dg - tg) ** 2).sum() 
                for dg, tg in zip(dummy_grads, true_gradients)
            )
            
            grad_diff.backward()
            optimizer.step()
            
            # Clamp to valid range
            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, 0, 1)
        
        print(f"Reconstruction complete")
        all_reconstructed.append(dummy_data.detach())
        all_labels.append(recovered_label)
    
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    return all_reconstructed, all_labels


def model_inversion_attack(model, device, num_classes=10):
    """
    Model Inversion: Generate class representatives by maximizing model confidence.
    """
    print("\n" + "#"*70)
    print("# ATTACK 2: MODEL INVERSION")
    print("#"*70)
    
    all_images = []
    
    for target_class in range(num_classes):
        print(f"\nGenerating class {target_class}...")
        
        # Initialize from dataset mean
        dummy_data = torch.full(
            (1, 1, 28, 28), 
            0.1307,
            device=device,
            requires_grad=True
        )
        
        # Multi-stage optimization
        stages = [
            (500, 1.0),
            (500, 0.1),
            (500, 0.01)
        ]
        
        for stage_idx, (iters, lr) in enumerate(stages):
            optimizer = torch.optim.Adam([dummy_data], lr=lr)
            
            for i in range(iters):
                optimizer.zero_grad()
                
                output = model(dummy_data)
                probs = F.softmax(output, dim=1)
                
                # Maximize confidence for target class
                confidence_loss = -torch.log(probs[0, target_class] + 1e-10)
                
                # Total variation for smoothness
                tv_h = torch.abs(dummy_data[:, :, 1:, :] - dummy_data[:, :, :-1, :]).sum()
                tv_w = torch.abs(dummy_data[:, :, :, 1:] - dummy_data[:, :, :, :-1]).sum()
                tv_loss = (tv_h + tv_w) / dummy_data.numel()
                
                # L2 regularization toward mean
                l2_loss = torch.norm(dummy_data - 0.1307)
                
                total_loss = confidence_loss + 0.0001 * tv_loss + 0.0001 * l2_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Clamp to valid range
                with torch.no_grad():
                    dummy_data.data = torch.clamp(dummy_data.data, 0, 1)
        
        # Final evaluation
        with torch.no_grad():
            output = model(dummy_data)
            probs = F.softmax(output, dim=1)
            confidence = probs[0, target_class].item()
            predicted = torch.argmax(probs).item()
            print(f"  ✓ Confidence: {confidence:.4f}, Predicted: {predicted}")
        
        all_images.append(dummy_data.detach())
    
    all_images = torch.cat(all_images, dim=0)
    return all_images


def main():
    print("\n" + "="*70)
    print(" COMPLETE MODEL INVERSION ATTACK SUITE")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model_path = 'global_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    model = load_model(model_path, device)
    grad_loader, gan_loader = prepare_data()
    os.makedirs('attack_results', exist_ok=True)
    
    # ========================================================================
    # ATTACK 1: Gradient Inversion
    # ========================================================================
    grad_images, grad_labels = gradient_inversion_attack(
        model, grad_loader, device, num_samples=10
    )
    
    visualize_all_classes(
        reconstructed_images=grad_images,
        num_classes=10,
        attack_name="Gradient Inversion (iDLG)",
        save_path='attack_results/gradient_inversion_idlg.png'
    )
    print(f"\n✓ Gradient inversion complete! Recovered labels: {grad_labels}")
    
    # ========================================================================
    # ATTACK 2: Model Inversion
    # ========================================================================
    model_inv_images = model_inversion_attack(model, device, num_classes=10)
    
    visualize_all_classes(
        reconstructed_images=model_inv_images,
        num_classes=10,
        attack_name="Model Inversion",
        save_path='attack_results/model_inversion_all.png'
    )
    print(f"\n✓ Model inversion complete!")
    
    # ========================================================================
    # ATTACK 3: GAN-based Inversion
    # ========================================================================
    print("\n" + "#"*70)
    print("# ATTACK 3: GAN-BASED INVERSION (GMI)")
    print("#"*70)
    
    gan_attack = GANInversionAttack(
        target_model=model,
        device=device,
        latent_dim=100,
        img_shape=(1, 28, 28)
    )
    
    print("\nTraining GAN...")
    gan_attack.train_gan(dataloader=gan_loader, epochs=30, lr=0.0002)
    
    print("\nGenerating class representatives...")
    gan_images, _ = gan_attack.attack_all_classes(num_classes=10)
    
    visualize_all_classes(
        reconstructed_images=gan_images,
        num_classes=10,
        attack_name="GAN-based Inversion (GMI)",
        save_path='attack_results/gan_inversion_all.png'
    )
    print(f"\n✓ GAN inversion complete!")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print(" ALL ATTACKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  1. gradient_inversion_idlg.png - 10 reconstructed samples")
    print("  2. model_inversion_all.png     - All 10 class representatives")
    print("  3. gan_inversion_all.png       - GAN-generated representatives")
    print("\nAll files saved in: attack_results/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
