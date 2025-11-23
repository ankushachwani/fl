"""
Model Inversion Attacks for Federated Learning

This module implements several well-known model inversion attacks:
1. Deep Leakage from Gradients (DLG)
2. Model Inversion Attack (Fredrikson et al.)
3. GAN-based Model Inversion (GMI)
"""

from .gradient_inversion import GradientInversionAttack
from .model_inversion import ModelInversionAttack
from .gan_inversion import GANInversionAttack

__all__ = [
    'GradientInversionAttack',
    'ModelInversionAttack',
    'GANInversionAttack',
]
