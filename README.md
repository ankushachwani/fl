# Federated Learning with Model Inversion Attacks

Implementation of federated learning for privacy research. Trains a model using federated averaging, then demonstrates three different attacks that can reconstruct training data from the shared model updates.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py --config standard

# Run all three attacks
python run_all_attacks.py
```

Results appear in `attack_results/` showing reconstructed images from each attack method.

## What's Implemented

**Federated Learning System**
- Multiple clients train locally on their own data
- Only model updates get shared with central server  
- Server aggregates updates using FedAvg algorithm
- Tested on MNIST digit classification

**Three Privacy Attacks**
1. **Gradient Inversion (iDLG)** - Recovers training samples from shared gradients
2. **Model Inversion** - Generates class representatives by maximizing model confidence
3. **GAN-based Inversion** - Uses a generator network to create realistic samples

All three attacks work on the trained federated model and show different ways training data can leak even when data isn't directly shared.

## Training the Model

Choose config based on how thorough you want to be:

```bash
# Quick test - 3 clients, 3 rounds
python main.py --config quick

# Standard - 5 clients, 10 rounds (recommended)
python main.py --config standard

# Extensive - 10 clients, 20 rounds
python main.py --config extensive
```

Training takes 5-15 minutes depending on config. Creates `global_model.pth` when done.

**Config Details:**
- Quick: 3 clients, 3 rounds, 2 local epochs
- Standard: 5 clients, 10 rounds, 5 local epochs  
- Extensive: 10 clients, 20 rounds, 10 local epochs

All use batch size 64 and learning rate 0.01.

## Data Distribution

Edit `config.py` to change how data splits across clients:

```python
'data_distribution': 'iid',  # uniform random split
# or
'data_distribution': 'non_iid',  # each client gets only 2 digit classes
```

Non-IID is more realistic since real clients typically have biased data.

## Running Attacks

The `run_all_attacks.py` script runs all three attacks in sequence:

```bash
python run_all_attacks.py
```

Takes about 20-30 minutes total. You'll see:
- Gradient Inversion attacking 10 samples
- Model Inversion generating all 10 digit classes
- GAN training then generating all 10 classes

### What Each Attack Does

**Gradient Inversion (iDLG)**
- Intercepts gradients from one training batch
- Uses analytical method to recover the label
- Optimizes dummy input until its gradients match the real ones
- Works best for batch size 1, can reconstruct nearly exact images

**Model Inversion**
- Starts from random noise for each digit class
- Runs gradient ascent to maximize model confidence for that class
- Uses regularization (total variation + L2) to keep images realistic
- Shows what the model "thinks" each digit should look like

**GAN-based Inversion**  
- Trains a generator network on MNIST data
- For each class, optimizes the latent code to maximize model confidence
- Generator produces realistic images that fool the classifier
- Slowest but produces most realistic results

### Results

Check `attack_results/` for three PNG files:
- `gradient_inversion_idlg.png` - 10 reconstructed samples with recovered labels
- `model_inversion_all.png` - One representative image per digit class (0-9)
- `gan_inversion_all.png` - GAN-generated representatives for each class

## Project Structure

```
fl/
├── main.py                  # Train federated learning model
├── run_all_attacks.py       # Run all three attacks
├── config.py               # Configuration settings
├── global_model.pth        # Trained model (created after training)
├── src/
│   ├── client.py          # Client training logic
│   ├── server.py          # Server aggregation logic
│   ├── model.py           # CNN architecture
│   ├── utils.py           # Data handling
│   └── attacks/
│       ├── gan_inversion.py    # GAN attack implementation
│       └── visualization.py    # Plot results
└── attack_results/         # Attack outputs (created when attacks run)
```

## Model Architecture

Simple CNN for MNIST:
- Conv layer: 1→32 channels, 3x3 kernel
- MaxPool 2x2
- Conv layer: 32→64 channels, 3x3 kernel  
- MaxPool 2x2
- FC layer: 1024→128
- Dropout 0.5
- FC layer: 128→10 (output)

Gets about 95-98% accuracy after training.

## How the Attacks Work

**Gradient Inversion** matches gradients. If you have gradients ∇W from a batch, find dummy data x' where gradients of x' equal ∇W. Minimize ||∇W' - ∇W||² by adjusting x'.

**Model Inversion** maximizes confidence. For class c, find input x that maximizes P(y=c|x) using gradient ascent. Add regularization so images don't look like noise.

**GAN Inversion** trains a generator G first, then optimizes latent code z so G(z) has high confidence for target class. Generator keeps images realistic.

## Expected Runtime

On regular CPU:
- Training (standard): ~10 minutes
- Gradient inversion: ~3 minutes  
- Model inversion: ~8 minutes
- GAN training + inversion: ~15 minutes
- Total: ~35 minutes start to finish

GPU speeds things up 3-5x.

## Papers Implemented

**Gradient Inversion:**  
"Deep Leakage from Gradients" by Zhu et al., NeurIPS 2019  
https://arxiv.org/abs/1906.08935

Improved version (iDLG): "iDLG: Improved Deep Leakage from Gradients" by Zhao et al., 2020  
https://arxiv.org/abs/2001.02610

**Model Inversion:**  
"Model Inversion Attacks that Exploit Confidence Information" by Fredrikson et al., CCS 2015  
https://dl.acm.org/doi/10.1145/2810103.2813677

**GAN-based:**  
"The Secret Revealer: Generative Model-Inversion Attacks" by Zhang et al., CVPR 2020  
https://arxiv.org/abs/1911.07135

## Why This Matters

Shows that federated learning isn't automatically private. Even though clients never share raw data, the model updates (gradients/weights) can leak information about training samples. This is the whole point of the research - understanding these privacy risks so we can build better defenses.
