"""
Federated Learning System

A modular implementation of Federated Learning for privacy-preserving
distributed machine learning.
"""

__version__ = "1.0.0"
__author__ = "FL Research Team"

from .model import SimpleCNN
from .client import FederatedClient
from .server import FederatedServer
from .utils import (
    partition_data_iid,
    partition_data_non_iid,
    visualize_data_distribution,
    plot_training_history,
    print_client_info
)

__all__ = [
    'SimpleCNN',
    'FederatedClient',
    'FederatedServer',
    'partition_data_iid',
    'partition_data_non_iid',
    'visualize_data_distribution',
    'plot_training_history',
    'print_client_info'
]
