# Chemical Density Surrogate Model

A modular machine learning framework for predicting liquid densities from Lennard-Jones potential parameters.

## Project Structure

```
ChemicalDensity/
├── dataset.csv              # Training data (SigC, SigH, EpsC, EpsH → density)
│
├── train_api.py             # Public API: Train single models
├── ensemble_api.py          # Public API: Train deep ensembles
├── tune_api.py              # Public API: Hyperparameter optimization
├── analyze.py               # Public API: Data and model analysis
│
├── models/                  # Model architectures
│   ├── __init__.py          # Factory function: create_model()
│   ├── activations.py       # Custom activations (SwiGLU, MLPBlock)
│   ├── mlp.py               # Multi-layer perceptron
│   ├── cnn.py               # CNN and MultiScaleCNN
│   └── lightgbm_model.py    # LightGBM wrapper
│
├── core/                    # Core utilities
│   ├── __init__.py
│   ├── config.py            # TrainingConfig dataclass
│   ├── data_loader.py       # Dataset loading and splitting
│   ├── trainer.py           # Training loop for all model types
│   └── utils.py             # Seeds, device, normalization helpers
│
├── training/                # Training workflows
│   ├── __init__.py
│   ├── simple.py            # SimpleTrainer for single models
│   └── ensemble.py          # EnsembleTrainer for uncertainty quantification
│
├── tuning/                  # Hyperparameter optimization
│   ├── __init__.py
│   ├── search_spaces.py     # Optuna search space definitions
│   ├── objective.py         # Training objective function
│   └── manager.py           # HyperparameterTuner class
│
└── analysis/                # Analysis and visualization
    ├── __init__.py
    ├── data_analysis.py     # Dataset exploration
    ├── model_analysis.py    # Model evaluation
    └── plotting.py          # Visualization utilities
```

## Quick Start

### 1. Train a Model

```bash
# Train MLP with defaults
python train_api.py --architecture mlp

# Train CNN with custom parameters
python train_api.py --architecture cnn --epochs 100 --batch-size 64

# Train with tuned hyperparameters
python train_api.py --architecture mlp --tuned-dir results_tuning/mlp_...
```

### 2. Hyperparameter Tuning

```bash
# Quick tune (10 trials, 20 epochs)
python tune_api.py --architecture mlp --preset minimal

# Standard tune (50 trials, 50 epochs)
python tune_api.py --architecture cnn --preset balanced

# Production tune
python tune_api.py --architecture lightgbm --trials 100 --epochs 100
```

### 3. Train Ensemble

```bash
# Homogeneous ensemble
python ensemble_api.py --architecture mlp --n-models 5

# Heterogeneous ensemble
python ensemble_api.py --architectures mlp cnn cnn_multiscale --n-models 2

# Ensemble with tuned configs
python ensemble_api.py --architecture mlp --tuned-dir results_tuning/mlp_...
```

### 4. Analyze Results

```bash
# Analyze dataset
python analyze.py data --path dataset.csv --report

# Analyze trained model
python analyze.py model --results-dir results_mlp/ --plot

# Compare models
python analyze.py compare --dirs results_mlp results_cnn --plot

# Analyze ensemble
python analyze.py ensemble --results-dir results_ensemble/
```

## Supported Architectures

| Architecture     | Description                             | Parameters                                       |
| ---------------- | --------------------------------------- | ------------------------------------------------ |
| `mlp`            | Multi-layer perceptron with SwiGLU/SiLU | `hidden_dims`, `use_swiglu`, `dropout_rate`      |
| `cnn`            | 1D CNN with residual connections        | `expansion_size`, `num_layers`, `kernel_size`    |
| `cnn_multiscale` | Multi-scale CNN (Inception-style)       | `expansion_size`, `num_scales`, `base_channels`  |
| `lightgbm`       | Gradient boosting machine               | `num_leaves`, `learning_rate`, `num_boost_round` |

## Programmatic Usage

```python
from models import create_model, AVAILABLE_ARCHITECTURES
from core import DataLoader, Trainer, TrainingConfig
from training import SimpleTrainer, EnsembleTrainer
from tuning import HyperparameterTuner
from analysis import DataAnalyzer, ModelAnalyzer, Plotter

# Create model
model = create_model('mlp', {'hidden_dims': [256, 64, 32]})

# Load data
loader = DataLoader('dataset.csv')
train_loader, val_loader, test_loader, stats = loader.load()

# Train
trainer = Trainer(model)
trainer.train(train_loader, val_loader, num_epochs=100)
results = trainer.test(test_loader)

# Analyze
analyzer = DataAnalyzer('dataset.csv')
summary = analyzer.get_summary()
```

## Requirements

```
torch>=2.0
numpy
pandas
matplotlib
optuna
lightgbm  # optional, for LightGBM architecture
```

## Dataset

The dataset contains molecular simulation results mapping Lennard-Jones potential parameters to liquid density:

- **Features**: SigC, SigH, EpsC, EpsH (4 parameters)
- **Target**: density (kg/m³)
- **Samples**: 6557

## License

MIT License
