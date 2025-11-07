"""
Configuration file for Federated Learning System

This file contains all configurable parameters for the FL system.
Modify these settings to customize the federated learning experiment.
"""

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================

FL_CONFIG = {
    # Number of simulated clients in the federated learning system
    # Each client represents a separate entity with its own private data
    'num_clients': 5,

    # Number of communication rounds between clients and server
    # In each round, clients train locally and send updates to server
    'num_rounds': 10,

    # Number of clients selected per round (for client sampling)
    # Set to num_clients to use all clients every round
    # Can be set lower to simulate partial client participation
    'clients_per_round': 5,

    # Data distribution strategy
    # 'iid': Independent and Identically Distributed (random split)
    # 'non_iid': Non-IID distribution (each client has data from specific classes)
    'data_distribution': 'iid',

    # For non-IID: number of classes each client should have
    # Only used when data_distribution = 'non_iid'
    # E.g., 2 means each client only has data from 2 out of 10 classes
    'classes_per_client': 2,
}


# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Number of local training epochs each client performs per round
    # Higher values = more local training before sending updates
    'local_epochs': 5,

    # Batch size for training
    # Larger batch size = faster training but more memory usage
    'batch_size': 64,

    # Learning rate for local model training
    # Controls how much to update model weights during training
    'learning_rate': 0.01,

    # Number of classes in the dataset (10 for MNIST: digits 0-9)
    'num_classes': 10,
}


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    # Dataset to use: 'MNIST', 'CIFAR10', etc.
    # Currently optimized for MNIST
    'dataset_name': 'MNIST',

    # Path to store downloaded datasets
    'data_path': './data',

    # Whether to download dataset if not present
    'download': True,
}


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

SYSTEM_CONFIG = {
    # Random seed for reproducibility
    # Set to None for random behavior
    'random_seed': 42,

    # Whether to use GPU if available
    'use_cuda': True,

    # Verbosity level (0: minimal, 1: normal, 2: detailed)
    'verbose': 1,

    # Whether to save visualizations
    'save_plots': True,

    # Whether to print detailed client information
    'print_client_info': True,
}


# ============================================================================
# EASY CONFIGURATION PRESETS
# ============================================================================

def get_quick_test_config():
    """
    Quick test configuration for fast experimentation.
    Use this for quick testing and debugging.
    """
    config = {
        'fl': FL_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'dataset': DATASET_CONFIG.copy(),
        'system': SYSTEM_CONFIG.copy(),
    }

    # Override for quick testing
    config['fl']['num_clients'] = 3
    config['fl']['num_rounds'] = 3
    config['fl']['clients_per_round'] = 3
    config['training']['local_epochs'] = 2

    return config


def get_standard_config():
    """
    Standard configuration for normal federated learning experiments.
    """
    return {
        'fl': FL_CONFIG,
        'training': TRAINING_CONFIG,
        'dataset': DATASET_CONFIG,
        'system': SYSTEM_CONFIG,
    }


def get_extensive_config():
    """
    Extensive configuration for comprehensive experiments.
    More clients, more rounds, more thorough training.
    """
    config = {
        'fl': FL_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'dataset': DATASET_CONFIG.copy(),
        'system': SYSTEM_CONFIG.copy(),
    }

    # Override for extensive experiments
    config['fl']['num_clients'] = 10
    config['fl']['num_rounds'] = 20
    config['fl']['clients_per_round'] = 10
    config['training']['local_epochs'] = 10

    return config


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config(config):
    """Print configuration in a readable format."""
    print("\n" + "="*60)
    print("FEDERATED LEARNING CONFIGURATION")
    print("="*60)

    for category, params in config.items():
        print(f"\n{category.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")


def validate_config(config):
    """
    Validate configuration parameters.

    Raises:
        ValueError: If configuration is invalid
    """
    fl_config = config['fl']
    training_config = config['training']

    # Validate number of clients
    if fl_config['num_clients'] < 1:
        raise ValueError("num_clients must be at least 1")

    # Validate clients per round
    if fl_config['clients_per_round'] > fl_config['num_clients']:
        raise ValueError("clients_per_round cannot exceed num_clients")

    # Validate data distribution
    valid_distributions = ['iid', 'non_iid']
    if fl_config['data_distribution'] not in valid_distributions:
        raise ValueError(f"data_distribution must be one of {valid_distributions}")

    # Validate training parameters
    if training_config['local_epochs'] < 1:
        raise ValueError("local_epochs must be at least 1")

    if training_config['batch_size'] < 1:
        raise ValueError("batch_size must be at least 1")

    if training_config['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")

    print("Configuration validated successfully!")
