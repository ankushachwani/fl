"""
Utility functions for Federated Learning
Includes data partitioning, visualization, and helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict


def partition_data_iid(dataset, num_clients):
    """
    Partition dataset into IID (Independent and Identically Distributed) subsets.

    Each client gets a random subset of data, maintaining similar distribution.

    Args:
        dataset: The complete dataset
        num_clients (int): Number of clients to partition data for

    Returns:
        dict: Maps client_id -> list of data indices
    """
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients

    # Shuffle indices
    indices = np.random.permutation(num_samples)

    # Partition indices
    client_data = {}
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client

        # Last client gets remaining samples
        if client_id == num_clients - 1:
            end_idx = num_samples

        client_data[client_id] = indices[start_idx:end_idx].tolist()

    return client_data


def partition_data_non_iid(dataset, num_clients, classes_per_client=2):
    """
    Partition dataset into Non-IID subsets.

    Each client gets data from only a subset of classes, simulating real-world
    scenarios where clients have specialized/biased data.

    Args:
        dataset: The complete dataset (must have targets attribute)
        num_clients (int): Number of clients
        classes_per_client (int): Number of classes each client should have

    Returns:
        dict: Maps client_id -> list of data indices
    """
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Shuffle indices within each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Assign classes to clients
    client_data = {i: [] for i in range(num_clients)}

    # Distribute classes to clients
    class_assignments = []
    for client_id in range(num_clients):
        # Randomly select classes for this client
        selected_classes = np.random.choice(
            num_classes,
            classes_per_client,
            replace=False
        )
        class_assignments.append(selected_classes)

    # Distribute data from assigned classes to clients
    for client_id, assigned_classes in enumerate(class_assignments):
        for class_label in assigned_classes:
            # Get indices for this class
            available_indices = class_indices[class_label]

            # Calculate how many samples this client gets from this class
            samples_per_class = len(available_indices) // sum(
                class_label in assignment for assignment in class_assignments
            )

            # Assign samples to this client
            client_data[client_id].extend(available_indices[:samples_per_class])

            # Remove assigned indices
            class_indices[class_label] = available_indices[samples_per_class:]

    return client_data


def visualize_data_distribution(client_data, dataset, num_clients):
    """
    Visualize how data is distributed across clients.

    Creates a bar chart showing the number of samples per client
    and a heatmap showing class distribution.

    Args:
        client_data (dict): Maps client_id -> list of data indices
        dataset: The dataset
        num_clients (int): Number of clients
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_labels = np.array(dataset.labels)
    else:
        print("Cannot visualize: dataset has no labels")
        return

    num_classes = len(np.unique(all_labels))

    # Count samples per client
    samples_per_client = [len(client_data[i]) for i in range(num_clients)]

    # Count class distribution per client
    class_distribution = np.zeros((num_clients, num_classes))
    for client_id in range(num_clients):
        client_labels = all_labels[client_data[client_id]]
        for label in client_labels:
            class_distribution[client_id, label] += 1

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Samples per client
    ax1.bar(range(num_clients), samples_per_client, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Data Distribution Across Clients')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Class distribution heatmap
    im = ax2.imshow(class_distribution.T, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Class Label')
    ax2.set_title('Class Distribution per Client')
    ax2.set_xticks(range(num_clients))
    ax2.set_yticks(range(num_classes))

    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Number of Samples')

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
    print("\nData distribution visualization saved as 'data_distribution.png'")
    plt.close()


def plot_training_history(server, save_path='training_history.png'):
    """
    Plot training history including test accuracy and loss over rounds.

    Args:
        server: FederatedServer object with training history
        save_path (str): Path to save the plot
    """
    history = server.get_training_history()

    rounds = range(1, len(history['test_accuracies']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Test Accuracy
    ax1.plot(rounds, history['test_accuracies'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Global Model Test Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # Plot 2: Test Loss
    ax2.plot(rounds, history['test_losses'], 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Global Model Test Loss')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved as '{save_path}'")
    plt.close()


def print_client_info(client_data, dataset):
    """
    Print detailed information about client data distribution.

    Args:
        client_data (dict): Maps client_id -> list of data indices
        dataset: The dataset
    """
    print("\n" + "="*60)
    print("CLIENT DATA DISTRIBUTION")
    print("="*60)

    # Get labels
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_labels = np.array(dataset.labels)
    else:
        print("Dataset has no labels - showing sample counts only")
        for client_id, indices in client_data.items():
            print(f"Client {client_id}: {len(indices)} samples")
        return

    for client_id, indices in client_data.items():
        client_labels = all_labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)

        print(f"\nClient {client_id}:")
        print(f"  Total samples: {len(indices)}")
        print(f"  Class distribution:")
        for label, count in zip(unique, counts):
            percentage = (count / len(indices)) * 100
            print(f"    Class {label}: {count} samples ({percentage:.1f}%)")

    print("\n" + "="*60)
