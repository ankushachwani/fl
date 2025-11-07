"""
Neural Network Model for Federated Learning
This module defines the CNN model used by all clients in the FL system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.

    Architecture:
    - 2 Convolutional layers with ReLU activation and max pooling
    - 2 Fully connected layers
    - Designed for MNIST (28x28 grayscale images)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28x1 -> 28x28x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14x32 -> 14x14x64

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After 2 pooling: 28->14->7
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward pass through the network."""
        # Conv layer 1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28x32 -> 14x14x32

        # Conv layer 2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # 14x14x64 -> 7x7x64

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_parameters(self):
        """Get model parameters as a list of tensors."""
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of tensors."""
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()
