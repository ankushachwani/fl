"""
Main script for running Federated Learning experiments

This script orchestrates the entire federated learning process:
1. Load and partition data across clients
2. Initialize server and clients
3. Run federated training rounds
4. Evaluate and visualize results

Usage:
    python main.py [--config CONFIG_TYPE]

    CONFIG_TYPE options:
    - quick: Fast test with 3 clients and 3 rounds
    - standard: Normal experiment (default)
    - extensive: Comprehensive experiment with more clients and rounds
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import argparse
import sys
import os

# Import FL components
from src.model import SimpleCNN
from src.client import FederatedClient
from src.server import FederatedServer
from src.utils import (
    partition_data_iid,
    partition_data_non_iid,
    visualize_data_distribution,
    plot_training_history,
    print_client_info
)
from config import (
    get_quick_test_config,
    get_standard_config,
    get_extensive_config,
    print_config,
    validate_config
)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"Random seed set to: {seed}")


def load_data(dataset_config):
    """
    Load and prepare the dataset.

    Args:
        dataset_config (dict): Dataset configuration parameters

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=dataset_config['data_path'],
        train=True,
        download=dataset_config['download'],
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=dataset_config['data_path'],
        train=False,
        download=dataset_config['download'],
        transform=transform
    )

    print(f"\nDataset: {dataset_config['dataset_name']}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def create_clients(model, train_dataset, client_data, config):
    """
    Create federated learning clients.

    Args:
        model: The neural network model
        train_dataset: Training dataset
        client_data (dict): Maps client_id -> data indices
        config (dict): Configuration parameters

    Returns:
        list: List of FederatedClient objects
    """
    print("\n" + "="*60)
    print("CREATING FEDERATED CLIENTS")
    print("="*60)

    num_clients = config['fl']['num_clients']
    clients = []

    # Merge training and system config for client initialization
    client_config = {**config['training'], **config['system']}

    for client_id in range(num_clients):
        client = FederatedClient(
            client_id=client_id,
            model=model,
            data_indices=client_data[client_id],
            dataset=train_dataset,
            config=client_config
        )
        clients.append(client)
        print(f"Created {client}")

    return clients


def run_federated_learning(clients, server, config):
    """
    Execute the federated learning training process.

    Args:
        clients (list): List of FederatedClient objects
        server: FederatedServer object
        config (dict): Configuration parameters

    Returns:
        FederatedServer: Server with training history
    """
    num_rounds = config['fl']['num_rounds']
    clients_per_round = config['fl']['clients_per_round']

    print("\n" + "="*60)
    print("STARTING FEDERATED LEARNING")
    print("="*60)
    print(f"Total rounds: {num_rounds}")
    print(f"Clients per round: {clients_per_round}/{len(clients)}")

    # Initial evaluation
    print("\n" + "="*60)
    print("Initial Model Evaluation")
    print("="*60)
    server.evaluate_global_model()

    # Federated training rounds
    for round_idx in range(num_rounds):
        # Select clients for this round (for now, use all clients)
        selected_clients = clients[:clients_per_round]

        # Step 1: Clients train locally
        print(f"\nClients training locally...")
        for client in selected_clients:
            # Get global model parameters
            global_params = server.get_global_parameters()

            # Train local model
            train_stats = client.train_local_model(global_params)

        # Step 2: Server aggregates client updates
        server.federated_averaging(selected_clients)

        # Step 3: Evaluate global model
        server.evaluate_global_model()

    return server


def main():
    """Main function to run federated learning experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning System')
    parser.add_argument(
        '--config',
        type=str,
        default='standard',
        choices=['quick', 'standard', 'extensive'],
        help='Configuration preset to use (default: standard)'
    )
    args = parser.parse_args()

    # Load configuration
    if args.config == 'quick':
        config = get_quick_test_config()
        print("\nUsing QUICK TEST configuration")
    elif args.config == 'extensive':
        config = get_extensive_config()
        print("\nUsing EXTENSIVE configuration")
    else:
        config = get_standard_config()
        print("\nUsing STANDARD configuration")

    # Print and validate configuration
    print_config(config)
    validate_config(config)

    # Set random seed
    set_random_seed(config['system']['random_seed'])

    # Load dataset
    train_dataset, test_dataset = load_data(config['dataset'])

    # Partition data across clients
    print("\n" + "="*60)
    print("PARTITIONING DATA ACROSS CLIENTS")
    print("="*60)

    if config['fl']['data_distribution'] == 'iid':
        print("Using IID (random) data distribution")
        client_data = partition_data_iid(train_dataset, config['fl']['num_clients'])
    else:
        print(f"Using Non-IID data distribution ({config['fl']['classes_per_client']} classes per client)")
        client_data = partition_data_non_iid(
            train_dataset,
            config['fl']['num_clients'],
            config['fl']['classes_per_client']
        )

    # Print client information
    if config['system']['print_client_info']:
        print_client_info(client_data, train_dataset)

    # Visualize data distribution
    if config['system']['save_plots']:
        visualize_data_distribution(client_data, train_dataset, config['fl']['num_clients'])

    # Initialize global model
    print("\n" + "="*60)
    print("INITIALIZING GLOBAL MODEL")
    print("="*60)

    global_model = SimpleCNN(num_classes=config['training']['num_classes'])
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model: SimpleCNN")
    print(f"Total parameters: {num_params:,}")

    # Create clients
    clients = create_clients(global_model, train_dataset, client_data, config)

    # Create server
    print("\n" + "="*60)
    print("INITIALIZING FEDERATED SERVER")
    print("="*60)

    server_config = {**config['training'], **config['system']}
    server = FederatedServer(global_model, test_dataset, server_config)
    print("Server initialized successfully")

    # Run federated learning
    server = run_federated_learning(clients, server, config)

    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    history = server.get_training_history()
    final_accuracy = history['test_accuracies'][-1]
    final_loss = history['test_losses'][-1]

    print(f"\nFinal Global Model Performance:")
    print(f"  Test Accuracy: {final_accuracy:.2f}%")
    print(f"  Test Loss: {final_loss:.4f}")

    # Save training history plot
    if config['system']['save_plots']:
        plot_training_history(server)

    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)

    # Summary of generated files
    print("\nGenerated files:")
    if os.path.exists('data_distribution.png'):
        print("  - data_distribution.png (client data distribution)")
    if os.path.exists('training_history.png'):
        print("  - training_history.png (training curves)")

    return server, clients


if __name__ == "__main__":
    try:
        server, clients = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
