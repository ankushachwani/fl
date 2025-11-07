"""
Federated Learning Client Implementation
Each client trains on its local private data and sends model updates to the server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy


class FederatedClient:
    """
    Represents a single client in the Federated Learning system.

    Each client:
    - Has its own private dataset (subset of the full dataset)
    - Trains a local model on its data
    - Sends model updates (weights) to the central server
    - Never shares raw data
    """

    def __init__(self, client_id, model, data_indices, dataset, config):
        """
        Initialize a federated learning client.

        Args:
            client_id (int): Unique identifier for this client
            model: The neural network model (shared architecture)
            data_indices (list): Indices of data samples assigned to this client
            dataset: The full dataset (client only accesses its indices)
            config (dict): Configuration parameters (learning rate, batch size, etc.)
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Create client's private dataset
        self.data_indices = data_indices
        self.dataset = Subset(dataset, data_indices)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )

        # Training configuration
        self.learning_rate = config['learning_rate']
        self.local_epochs = config['local_epochs']

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # Track training statistics
        self.train_losses = []
        self.train_accuracies = []

    def train_local_model(self, global_parameters=None):
        """
        Train the model on local data for a specified number of epochs.

        Args:
            global_parameters (list, optional): Updated global model parameters from server

        Returns:
            dict: Training statistics (loss, accuracy)
        """
        # Update local model with global parameters if provided
        if global_parameters is not None:
            self.model.set_parameters(global_parameters)

        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(self.local_epochs):
            batch_losses = []
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Track statistics
                batch_losses.append(loss.item())
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            # Calculate epoch statistics
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_accuracy = 100. * correct / total

            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

        # Store statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)

        return {
            'client_id': self.client_id,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'num_samples': len(self.dataset)
        }

    def get_model_parameters(self):
        """
        Get the current model parameters.

        Returns:
            list: Model parameters (weights and biases)
        """
        return self.model.get_parameters()

    def get_num_samples(self):
        """Get the number of data samples this client has."""
        return len(self.dataset)

    def __str__(self):
        return f"Client {self.client_id} (samples: {len(self.dataset)})"
