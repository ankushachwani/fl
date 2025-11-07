"""
Federated Learning Server Implementation
The server coordinates training, aggregates client model updates, and maintains the global model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np


class FederatedServer:
    """
    Central server for Federated Learning.

    The server:
    - Maintains the global model
    - Coordinates training rounds with clients
    - Aggregates client model updates using Federated Averaging (FedAvg)
    - Evaluates global model performance
    - Never accesses client's raw data
    """

    def __init__(self, model, test_dataset, config):
        """
        Initialize the federated learning server.

        Args:
            model: The global neural network model
            test_dataset: Test dataset for evaluating global model
            config (dict): Configuration parameters
        """
        self.global_model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)

        # Test data loader for evaluation
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Track training history
        self.round_history = []
        self.test_accuracies = []
        self.test_losses = []

        self.current_round = 0

    def aggregate_parameters(self, client_parameters, client_weights):
        """
        Aggregate client model parameters using Federated Averaging (FedAvg).

        FedAvg computes weighted average of client models based on their dataset sizes.

        Args:
            client_parameters (list): List of parameter lists from each client
            client_weights (list): List of weights (typically proportional to dataset size)

        Returns:
            list: Aggregated global model parameters
        """
        # Normalize weights to sum to 1
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated parameters with zeros
        aggregated_params = []

        # Get number of parameter tensors
        num_params = len(client_parameters[0])

        # Aggregate each parameter tensor
        for param_idx in range(num_params):
            # Weighted sum of this parameter across all clients
            weighted_param = torch.zeros_like(client_parameters[0][param_idx])

            for client_idx, params in enumerate(client_parameters):
                weighted_param += normalized_weights[client_idx] * params[param_idx]

            aggregated_params.append(weighted_param)

        return aggregated_params

    def federated_averaging(self, clients):
        """
        Perform one round of Federated Averaging.

        Steps:
        1. Collect model parameters from all clients
        2. Calculate weights based on client dataset sizes
        3. Aggregate parameters using weighted averaging
        4. Update global model

        Args:
            clients (list): List of FederatedClient objects

        Returns:
            dict: Round statistics
        """
        print(f"\n{'='*60}")
        print(f"Round {self.current_round + 1}")
        print(f"{'='*60}")

        # Collect parameters and weights from all clients
        client_parameters = []
        client_weights = []
        round_stats = []

        for client in clients:
            # Get client's model parameters
            params = client.get_model_parameters()
            client_parameters.append(params)

            # Weight by dataset size (larger datasets have more influence)
            num_samples = client.get_num_samples()
            client_weights.append(num_samples)

            # Collect training stats
            if len(client.train_losses) > 0:
                round_stats.append({
                    'client_id': client.client_id,
                    'loss': client.train_losses[-1],
                    'accuracy': client.train_accuracies[-1],
                    'num_samples': num_samples
                })

        # Aggregate parameters using FedAvg
        aggregated_params = self.aggregate_parameters(client_parameters, client_weights)

        # Update global model
        self.global_model.set_parameters(aggregated_params)

        # Print client statistics
        print(f"\nClient Training Results:")
        for stats in round_stats:
            print(f"  Client {stats['client_id']}: "
                  f"Loss={stats['loss']:.4f}, "
                  f"Accuracy={stats['accuracy']:.2f}%, "
                  f"Samples={stats['num_samples']}")

        # Store round history
        self.round_history.append({
            'round': self.current_round,
            'client_stats': round_stats,
            'num_clients': len(clients)
        })

        self.current_round += 1

        return {
            'round': self.current_round,
            'client_stats': round_stats
        }

    def evaluate_global_model(self):
        """
        Evaluate the global model on the test dataset.

        Returns:
            dict: Test loss and accuracy
        """
        self.global_model.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.global_model(data)
                loss = self.criterion(output, target)

                # Accumulate loss
                test_loss += loss.item()

                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        # Calculate averages
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        # Store history
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)

        print(f"\nGlobal Model Evaluation:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }

    def get_global_parameters(self):
        """Get the current global model parameters."""
        return self.global_model.get_parameters()

    def get_training_history(self):
        """
        Get complete training history.

        Returns:
            dict: Training and evaluation history
        """
        return {
            'rounds': self.round_history,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses
        }
