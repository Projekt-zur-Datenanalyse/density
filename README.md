# Chemical Density Surrogate Model

A modular, configurable PyTorch-based surrogate model for predicting chemical density from molecular features (SigC, SigH, EpsC, EpsH).

## Overview

This project implements a flexible MLP-based surrogate model with the following features:

- **Configurable Architecture**: Control number of layers, hidden dimensions, and activation functions
- **SwiGLU Activation**: State-of-the-art gated linear unit activation (default) or simple Swish
- **Residual Connections**: Automatic residual connections for multi-layer networks (num_layers > 1)
- **Full CUDA Support**: Seamless GPU acceleration with PyTorch
- **Modular Design**: Easy to extend and modify for other applications

## Project Structure

```
.
├── config.py                  # Configuration classes
├── activation.py              # Activation functions (Swish, SwiGLU)
├── model.py                   # Core model architecture
├── data_loader.py             # Data loading and preprocessing
├── trainer.py                 # Training utilities
├── train.py                   # Main training script
├── examples.py                # Usage examples
├── README.md                  # This file
├── Dataset_1.csv              # Training data
├── Dataset_2.csv              # Training data
├── Dataset_3.csv              # Training data
├── Dataset_4.csv              # Training data
└── results/                   # Output directory (created automatically)
    ├── checkpoints/           # Model checkpoints
    ├── model_config.json      # Model configuration
    ├── training_config.json   # Training configuration
    ├── normalization_stats.json # Data normalization statistics
    ├── training_history.json  # Training loss history
    ├── test_results.json      # Test metrics
    └── predictions.pt         # Test predictions
```

## Architecture

### Model Components

**Input Layer**: 4 features

- SigC (sigma carbon)
- SigH (sigma hydrogen)
- EpsC (epsilon carbon)
- EpsH (epsilon hydrogen)

**Hidden Layers**: Configurable number of MLP blocks

- Default: 1 layer
- Each block can use SwiGLU or Swish activation
- Residual connections connect consecutive layers (when num_layers > 1)

**Output Layer**: Single neuron predicting density

### Example Configurations

#### Default (Single Layer, SwiGLU)

```python
config = ModelConfig(
    num_layers=1,           # Single hidden layer
    expansion_factor=100,   # 4 * 100 = 400 hidden units
    use_swiglu=True,        # SwiGLU activation
)
```

#### Multi-Layer with Residuals

```python
config = ModelConfig(
    num_layers=3,           # 3 hidden layers with residuals
    expansion_factor=100,
    use_swiglu=True,
)
```

#### Swish Activation Only

```python
config = ModelConfig(
    num_layers=2,
    expansion_factor=100,
    use_swiglu=False,       # Use Swish instead of SwiGLU
)
```

## Usage

### 1. Basic Model Creation

```python
from config import ModelConfig
from model import ChemicalDensitySurrogate

# Create configuration
config = ModelConfig(
    num_layers=1,
    expansion_factor=100,
    use_swiglu=True,
)

# Create model
model = ChemicalDensitySurrogate(config)

# Print model info
print(model.get_model_info())
```

### 2. Making Predictions

```python
import torch

# Create sample input: batch_size=8, features=4
input_features = torch.randn(8, 4)

# Forward pass
densities = model(input_features)  # Shape: (8, 1)
```

### 3. Training

#### Quick Start

```bash
python train.py
```

#### Custom Configuration

```bash
python train.py \
    --num-layers 2 \
    --expansion-factor 100 \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --optimizer adam \
    --loss mse \
    --device cuda
```

#### Available Arguments

**Model Configuration**:

- `--num-layers`: Number of hidden layers (default: 1)
- `--expansion-factor`: Hidden dimension multiplier (default: 100)
- `--use-swiglu`: Use SwiGLU activation (default flag)
- `--use-swish`: Use Swish activation instead of SwiGLU

**Training Configuration**:

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--learning-rate`: Learning rate (default: 0.001)
- `--optimizer`: adam or sgd (default: adam)
- `--loss`: mse or mae (default: mse)

**Data Configuration**:

- `--data-dir`: Directory containing CSV files (default: current directory)
- `--val-split`: Validation split ratio (default: 0.2)
- `--test-split`: Test split ratio (default: 0.1)
- `--no-normalize`: Disable input/target normalization

**General**:

- `--device`: cuda or cpu (auto-detected by default)
- `--seed`: Random seed (default: 42)
- `--output-dir`: Results directory (default: ./results)

### 4. Examples

Run all example scripts:

```bash
python examples.py
```

This demonstrates:

1. Basic model creation and forward pass
2. Model with Swish activation
3. Multi-layer model with residuals
4. Comparing different expansion factors
5. CUDA device placement
6. Batch predictions

## Configuration Files

### ModelConfig

```python
@dataclass
class ModelConfig:
    input_dim: int = 4                  # Fixed at 4 features
    output_dim: int = 1                 # Single density output
    num_layers: int = 1                 # Number of hidden layers
    expansion_factor: float = 100       # Multiplier for hidden dimension
    use_swiglu: bool = True             # SwiGLU vs Swish activation
    device: str = "cuda"                # Device placement
    dtype: str = "float32"              # Data type
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    loss_fn: str = "mse"                # mse or mae
    optimizer: str = "adam"             # adam or sgd
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
```

## Activation Functions

### Swish

Simple smooth activation function:
$$\text{Swish}(x) = x \cdot \sigma(x)$$

### SwiGLU

Gated linear unit with Swish activation:
$$\text{SwiGLU}(x) = (x \cdot W + b) \odot \text{Swish}(x \cdot V + c)$$

Where $\odot$ denotes element-wise multiplication.

**References**: "GLU Variants Improve Transformer" (Shazeer et al., 2020)

## Data Format

The model expects CSV files with the following structure:

```
,Feature1,Feature2,...,FeatureN
SigC,value1,value2,...,valueN
SigH,value1,value2,...,valueN
EpsC,value1,value2,...,valueN
EpsH,value1,value2,...,valueN
Density,value1,value2,...,valueN
```

Each column (except the first) represents a sample with 4 input features and 1 target (density).

## Training Process

1. **Data Loading**: Loads all Dataset\_\*.csv files, combines them, and splits into train/val/test sets
2. **Normalization**: Optionally normalizes features and targets to zero mean and unit variance
3. **Model Creation**: Instantiates the surrogate model with specified configuration
4. **Training Loop**:
   - For each epoch:
     - Train on training set with backpropagation
     - Validate on validation set
     - Save best model based on validation loss
5. **Testing**: Evaluate final model on held-out test set
6. **Results**: Save model, configurations, predictions, and training history

## Output Files

After training, the `results/` directory contains:

- `best_model.pt` - Best model weights and configuration
- `model_config.json` - Model architecture configuration
- `training_config.json` - Training hyperparameters
- `normalization_stats.json` - Feature and target normalization statistics
- `training_history.json` - Loss history for train/val sets
- `test_results.json` - Test set metrics
- `predictions.pt` - Model predictions and targets on test set
- `checkpoints/` - Periodic checkpoints during training

## Performance Considerations

### Model Size

- **Expansion Factor 50**: ~200 parameters (1 layer)
- **Expansion Factor 100**: ~400 parameters (1 layer) ← **Recommended for default**
- **Expansion Factor 200**: ~800 parameters (1 layer)

### Computation

- **Single Layer**: Very fast, suitable for real-time predictions
- **Multi-Layer with Residuals**: ~2-3x slower per layer, better expressiveness
- **SwiGLU vs Swish**: SwiGLU ~2x more parameters due to gating mechanism

### Memory

- **Batch Size 64**: ~1 MB (GPU memory)
- **Batch Size 256**: ~4 MB (GPU memory)

## Requirements

- PyTorch >= 1.9.0
- CUDA 11.x or higher (for GPU acceleration)
- NumPy
- Pandas
- Python 3.8+

## Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas
```

## Extending the Model

### Adding Custom Activation Functions

1. Create a new class in `activation.py`:

```python
class CustomActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your activation logic
        return x
```

2. Update `MLPBlock` to support it

### Custom Model Architectures

Inherit from the base model:

```python
class CustomDensityModel(ChemicalDensitySurrogate):
    def _build_network(self):
        # Your custom layer structure
        pass
```

### Adding Regularization

Modify `MLPBlock` to add dropout or other regularization techniques.

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Reduce `--expansion-factor`
- Use `--device cpu` (slower but less memory)

### Poor Performance

- Increase `--epochs`
- Adjust `--learning-rate`
- Try `--optimizer sgd`
- Disable `--no-normalize` if data is already normalized

### Device Issues

Verify CUDA installation:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Citation

If you use this model in your research, please cite:

```bibtex
@software{chemical_density_surrogate,
  title={Chemical Density Surrogate Model},
  author={Your Name},
  year={2024}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please open an issue or contact [your contact info].
