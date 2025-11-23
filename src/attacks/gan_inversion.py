"""
GAN-based Model Inversion (GMI) Attack

Paper: "The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks"
by Zhang et al. (CVPR 2020)
Link: https://arxiv.org/abs/1911.07135

This attack uses a GAN (Generative Adversarial Network) to generate realistic
images that the target model classifies with high confidence. The GAN is trained
to fool the target classifier while producing realistic images.

Process:
1. Train a GAN (Generator + Discriminator) on auxiliary data
2. Given a target class, optimize the GAN's latent code to:
   - Maximize the target model's confidence for that class
   - Generate realistic images (discriminator constraint)
3. The generator produces class representatives that look realistic

Note: For MNIST, this is simplified as the images are simple.
For complex datasets (faces, etc.), this is more powerful than direct optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import numpy as np


class Generator(nn.Module):
    """Simple GAN Generator for MNIST (28x28 grayscale images)"""
    
    def __init__(self, latent_dim: int = 100, img_shape: Tuple = (1, 28, 28)):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.img_size),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """Simple GAN Discriminator for MNIST"""
    
    def __init__(self, img_shape: Tuple = (1, 28, 28)):
        super(Discriminator, self).__init__()
        self.img_size = int(np.prod(img_shape))
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class GANInversionAttack:
    """
    Implements GAN-based Model Inversion Attack (Zhang et al., 2020).
    
    Uses a GAN to generate realistic-looking images that maximize
    the target model's confidence for specific classes.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        device: str = 'cpu',
        latent_dim: int = 100,
        img_shape: Tuple = (1, 28, 28)
    ):
        """
        Args:
            target_model: The target neural network to attack
            device: Device to run on ('cpu' or 'cuda')
            latent_dim: Dimension of GAN's latent space
            img_shape: Shape of generated images
        """
        self.target_model = target_model
        self.device = device
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        # Initialize GAN
        self.generator = Generator(latent_dim, img_shape).to(device)
        self.discriminator = Discriminator(img_shape).to(device)
        
        self.gan_trained = False
        
    def train_gan(
        self,
        dataloader,
        epochs: int = 50,
        lr: float = 0.0002,
        beta1: float = 0.5
    ):
        """
        Pre-train the GAN on auxiliary data.
        
        Args:
            dataloader: DataLoader with training images
            epochs: Number of training epochs
            lr: Learning rate
            beta1: Adam optimizer beta1 parameter
        """
        print(f"\n{'='*60}")
        print("TRAINING GAN FOR MODEL INVERSION")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"{'='*60}\n")
        
        # Loss function
        adversarial_loss = nn.BCELoss()
        
        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):
                batch_size = imgs.size(0)
                imgs = imgs.to(self.device)
                
                # Normalize to [-1, 1] to match Tanh generator output
                imgs = imgs * 2 - 1
                
                # Adversarial ground truths
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                
                # ---------------------
                #  Train Generator
                # ---------------------
                optimizer_G.zero_grad()
                
                # Sample noise
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # Generate images
                gen_imgs = self.generator(z)
                
                # Loss: fool discriminator
                g_loss = adversarial_loss(self.discriminator(gen_imgs), valid)
                
                g_loss.backward()
                optimizer_G.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                
                # Measure discriminator's ability to classify real/fake
                real_loss = adversarial_loss(self.discriminator(imgs), valid)
                fake_loss = adversarial_loss(
                    self.discriminator(gen_imgs.detach()),
                    fake
                )
                d_loss = (real_loss + fake_loss) / 2
                
                d_loss.backward()
                optimizer_D.step()
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"D Loss: {d_loss.item():.4f} | "
                      f"G Loss: {g_loss.item():.4f}")
        
        self.gan_trained = True
        print("\nGAN training completed!")
        
    def attack(
        self,
        target_class: int,
        iterations: int = 300,
        lr: float = 0.01,
        confidence_weight: float = 1.0,
        realism_weight: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Execute GAN-based inversion attack for a target class.
        
        Args:
            target_class: Class label to reconstruct
            iterations: Number of optimization iterations
            lr: Learning rate for latent code optimization
            confidence_weight: Weight for classification loss
            realism_weight: Weight for discriminator (realism) loss
            
        Returns:
            Tuple of (reconstructed_image, latent_code, loss_history)
        """
        if not self.gan_trained:
            print("Warning: GAN not trained! Results may be poor.")
        
        self.generator.eval()
        self.target_model.eval()
        
        # Initialize latent code
        z = torch.randn(
            1, self.latent_dim,
            device=self.device
        )
        z.requires_grad = True
        
        optimizer = torch.optim.Adam([z], lr=lr)
        target_label = torch.tensor([target_class], device=self.device)
        
        history = []
        
        print(f"\n{'='*60}")
        print("GAN-BASED MODEL INVERSION ATTACK")
        print(f"{'='*60}")
        print(f"Target class: {target_class}")
        print(f"Iterations: {iterations}")
        print(f"Confidence weight: {confidence_weight}")
        print(f"Realism weight: {realism_weight}")
        print(f"{'='*60}\n")
        
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            # Generate image from latent code
            gen_img = self.generator(z)
            
            # Convert from [-1, 1] to [0, 1] for target model
            gen_img_normalized = (gen_img + 1) / 2
            gen_img_normalized = torch.clamp(gen_img_normalized, 0, 1)
            
            # Classification loss: maximize confidence for target class
            output = self.target_model(gen_img_normalized)
            confidence_loss = -F.log_softmax(output, dim=1)[0, target_class]
            
            # Realism loss: fool discriminator (want it to say "real")
            realism_score = self.discriminator(gen_img)
            realism_loss = -torch.log(realism_score + 1e-10)
            
            # Total loss
            total_loss = (
                confidence_weight * confidence_loss +
                realism_weight * realism_loss.mean()
            )
            
            total_loss.backward()
            optimizer.step()
            
            history.append({
                'total_loss': total_loss.item(),
                'confidence_loss': confidence_loss.item(),
                'realism_loss': realism_loss.mean().item()
            })
            
            if iteration % 50 == 0 or iteration == iterations - 1:
                with torch.no_grad():
                    pred_probs = F.softmax(output, dim=1)
                    pred_class = torch.argmax(pred_probs, dim=1).item()
                    target_confidence = pred_probs[0, target_class].item()
                    realism = realism_score.mean().item()
                
                print(f"Iteration {iteration:4d} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Confidence: {target_confidence:.4f} | "
                      f"Realism: {realism:.4f} | "
                      f"Pred: {pred_class}")
        
        # Generate final image
        with torch.no_grad():
            reconstructed_img = self.generator(z)
            reconstructed_img = (reconstructed_img + 1) / 2  # Denormalize
            reconstructed_img = torch.clamp(reconstructed_img, 0, 1)
        
        print(f"\nAttack completed!")
        
        return reconstructed_img, z.detach(), history
    
    def attack_all_classes(
        self,
        num_classes: int = 10,
        iterations: int = 300
    ) -> Tuple[torch.Tensor, Dict[int, List[float]]]:
        """
        Execute GAN attack for all classes.
        
        Args:
            num_classes: Number of classes
            iterations: Optimization iterations per class
            
        Returns:
            Tuple of (all_reconstructions, all_histories)
        """
        all_reconstructions = []
        all_histories = {}
        
        print(f"\n{'='*60}")
        print(f"ATTACKING ALL {num_classes} CLASSES WITH GAN")
        print(f"{'='*60}\n")
        
        for target_class in range(num_classes):
            reconstructed, _, history = self.attack(
                target_class=target_class,
                iterations=iterations
            )
            all_reconstructions.append(reconstructed)
            all_histories[target_class] = history
        
        all_reconstructions = torch.cat(all_reconstructions, dim=0)
        
        return all_reconstructions, all_histories
