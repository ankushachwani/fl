# Federated Learning System

Basic implementation of federated learning for model inversion attack research. Built from scratch using PyTorch.

## What This Does

Simulates multiple clients training a shared model without directly sharing their data. Each client trains locally, then sends only model weights to a central server. The server aggregates these weights and sends back an updated global model.

Uses MNIST digit classification as the test case.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running It

Three config options depending on how many clients are required:

```bash
# Quick test - 3 clients, 3 rounds
python main.py --config quick

# Normal run - 5 clients, 10 rounds 
python main.py --config standard

# Full experiment - 10 clients, 20 rounds
python main.py --config extensive
```

## Configuration Details

You can tweak settings in `config.py`, but here are the defaults:

**Quick Test:**
- Clients: 3
- Communication rounds: 3
- Local epochs per round: 2
- Batch size: 64
- Learning rate: 0.01

**Standard (default):**
- Clients: 5
- Communication rounds: 10
- Local epochs per round: 5
- Batch size: 64
- Learning rate: 0.01

**Extensive:**
- Clients: 10
- Communication rounds: 20
- Local epochs per round: 10
- Batch size: 64
- Learning rate: 0.01

## Data Distribution

You can simulate two scenarios in `config.py`:

**IID (default):** Data is randomly split across clients. Each client has similar data distribution.

**Non-IID:** Each client only gets data from 2 (out of 10) digit classes. More realistic - simulates clients with biased/specialized data.

Change this:
```python
'data_distribution': 'iid',  # or 'non_iid'
```

## What You Get

After running, you'll see:
- Console output showing training progress for each round
- `data_distribution.png` - visualization of how data is split across clients
- `training_history.png` - accuracy and loss curves over training rounds

Expected accuracy: ~95-98% after 10 rounds on MNIST.

## Project Structure

```
.
├── main.py              # Main script to run experiments
├── config.py            # All configuration parameters
├── requirements.txt     # Python dependencies
├── data/
│   └── MNIST/          # Dataset (auto-downloaded)
└── src/
    ├── client.py       # FederatedClient class
    ├── server.py       # FederatedServer class
    ├── model.py        # SimpleCNN model architecture
    └── utils.py        # Data partitioning and visualization
```

## Model Architecture

Simple CNN with:
- 2 convolutional layers (32 and 64 filters)
- 2 max pooling layers
- 2 fully connected layers (128 hidden units)
- Dropout for regularization

## Next Steps

This is the baseline. The plan is to implement model inversion attacks to see if we can reconstruct training images from the shared model weights. 
