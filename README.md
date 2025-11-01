# Chemical Density Surrogate Model

A modular, configurable PyTorch-based surrogate model for predicting chemical density from molecular features (SigC, SigH, EpsC, EpsH).

## Overview

This project implements multiple deep learning architectures for chemical density prediction with the following features:

- **Multiple Architectures**: Support for MLP, CNN, Multi-Scale CNN, and Graph Neural Networks (GNN)
- **Architecture-Agnostic Framework**: Unified training pipeline supporting all models
- **Configuration-Based Selection**: Switch architectures via `config.py` or CLI arguments
- **Comprehensive Benchmarking**: Compare all architectures with detailed metrics and visualizations
- **Full CUDA Support**: Seamless GPU acceleration with PyTorch
- **Modular Design**: Easy to extend with new architectures and components

## Project Structure

```
.
├── config.py                          # Configuration classes for all architectures
├── activation.py                      # Activation functions (Swish, SwiGLU)
├── model.py                           # Core model architectures
├── models/
│   ├── mlp.py                         # Multi-Layer Perceptron architecture
│   ├── cnn.py                         # Convolutional Neural Network
│   ├── cnn_multiscale.py              # Multi-Scale CNN with residual connections
│   └── gnn.py                         # Graph Neural Network (GAT/GCN)
├── data_loader.py                     # Data loading and preprocessing
├── trainer.py                         # Training utilities and loop
├── train.py                           # Main training script (architecture-agnostic)
├── benchmark_all_architectures.py     # Multi-architecture benchmarking script
├── analyze_data_ranges.py             # Data analysis utility
├── analyze_rmse.py                    # RMSE interpretation and metrics
├── PERFORMANCE_ANALYSIS.md            # Detailed performance documentation
├── examples.py                        # Usage examples
├── README.md                          # This file
├── Dataset_1.csv                      # Training data
├── Dataset_2.csv                      # Training data
├── Dataset_3.csv                      # Training data
├── Dataset_4.csv                      # Training data
├── results_mlp/                       # Output directory for MLP (created automatically)
├── results_cnn/                       # Output directory for CNN (created automatically)
├── results_cnn_multiscale/            # Output directory for Multi-Scale CNN (created automatically)
├── results_gnn/                       # Output directory for GNN (created automatically)
├── benchmark_results.png              # Comparative benchmark visualization
├── benchmark_results.json             # Benchmark metrics in JSON format
└── .gitignore
```

Each results directory contains:

- `best_model.pt` - Best model weights and configuration
- `model_config.json` - Model architecture configuration
- `training_config.json` - Training hyperparameters
- `normalization_stats.json` - Feature and target normalization statistics
- `training_history.json` - Loss history for train/val sets
- `test_results.json` - Test set metrics
- `predictions.pt` - Model predictions and targets on test set

## Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA 11.x or higher (optional, for GPU acceleration)

### Step 1: Clone Repository

```bash
git clone https://github.com/Projekt-zur-Datenanalyse/density.git
cd density
```

### Step 2: Create Virtual Environment

#### On Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: Install PyTorch CPU-only (if you don't have CUDA)
# pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy pandas matplotlib scikit-learn
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Architecture Overview

This project supports four distinct deep learning architectures for density prediction:

### 1. Multi-Layer Perceptron (MLP)

**Description**: Traditional feed-forward neural network with fully connected layers.

**Key Features**:

- Simple and fast
- Good baseline for comparison
- Configurable number of layers and hidden dimensions
- Optional SwiGLU gating mechanism

**Best for**: Quick prototyping, low-latency inference

**Configuration**:

```python
architecture: "mlp"
num_layers: 1
expansion_factor: 8  # Hidden dim = 4 * 8 = 32
use_swiglu: true
```

### 2. Convolutional Neural Network (CNN)

**Description**: 1D convolutional layers designed to capture local feature patterns.

**Key Features**:

- Efficient feature extraction via convolutions
- Configurable kernel size and number of layers
- Batch normalization support
- Optional residual connections

**Best for**: Capturing local correlations between features

**Configuration**:

```python
architecture: "cnn"
cnn_num_layers: 4
cnn_kernel_size: 3
cnn_expansion_size: 8
cnn_use_batch_norm: true
cnn_use_residual: false
```

### 3. Multi-Scale CNN

**Description**: CNN with multiple parallel branches operating at different scales, combined via late fusion.

**Key Features**:

- Multi-resolution feature extraction
- Parallel processing of multiple kernel sizes (3, 5, 7)
- Residual connections between scales
- Batch normalization
- More expressive than single-scale CNN

**Best for**: Complex feature interactions at multiple scales

**Configuration**:

```python
architecture: "cnn_multiscale"
cnn_multiscale_num_scales: 3
cnn_multiscale_base_channels: 16
cnn_multiscale_expansion_size: 8
num_layers: 4
cnn_use_residual: true
```

### 4. Graph Neural Network (GNN)

**Description**: Treats features as node attributes in a fully connected graph, using graph convolution or graph attention.

**Key Features**:

- Models feature relationships as a graph
- Graph Attention Networks (GAT) or Graph Convolutional Networks (GCN)
- Learns node embeddings and attention weights
- Permutation-invariant architecture
- State-of-the-art for structured feature interactions

**Best for**: Learning complex feature relationships and interactions

**Configuration**:

```python
architecture: "gnn"
gnn_type: "gat"  # or "gcn"
gnn_hidden_dim: 64
gnn_num_layers: 3
dropout_rate: 0.2
```

### Performance Comparison (20 epochs, LR=0.01)

Based on benchmark results with 7,793 chemical compounds:

| Rank | Architecture         | Test RMSE    | Improvement | Parameters |
| ---- | -------------------- | ------------ | ----------- | ---------- |
| 🥇   | Graph Neural Network | 115.79 kg/m³ | 42.5%       | ~25K       |
| 🥈   | Multi-Scale CNN      | 132.11 kg/m³ | 34.4%       | ~40K       |
| 🥉   | CNN                  | 135.53 kg/m³ | 32.8%       | ~16K       |
| 4️⃣   | MLP                  | 141.32 kg/m³ | 29.8%       | ~6K        |

**Baseline**: Naive prediction (always predict mean) = 201.44 kg/m³ RMSE

## Usage

### 1. Quick Start - Training a Model

#### Train Default MLP (Quick)

```bash
python train.py --epochs 50
```

#### Train GNN (Best Performance)

```bash
python train.py --architecture gnn --epochs 100
```

#### Train Multi-Scale CNN

```bash
python train.py --architecture cnn_multiscale --epochs 100
```

### 2. Configuration-Based Training

Edit `config.py` to set your preferred defaults:

```python
@dataclass
class ModelConfig:
    architecture: str = "gnn"           # Choose: "mlp", "cnn", "cnn_multiscale", "gnn"
    # ... rest of architecture-specific parameters
```

Then simply run:

```bash
python train.py --epochs 100
```

### 3. Training with Custom Hyperparameters

```bash
python train.py \
    --architecture gnn \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.005 \
    --device cuda
```

### 4. Making Predictions

```python
import torch
from config import ModelConfig
from models.gnn import GraphNeuralNetwork

# Load trained model
config = ModelConfig(architecture="gnn")
model = GraphNeuralNetwork(config)
checkpoint = torch.load("results_gnn/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
input_features = torch.randn(8, 4)  # Batch of 8 samples, 4 features
with torch.no_grad():
    predictions = model(input_features)
```

### 5. Complete Benchmarking (Compare All Architectures)

Run full benchmark comparing all 4 architectures:

```bash
python benchmark_all_architectures.py --epochs 100
```

This generates:

- **benchmark_results.png**: 6-panel visualization comparing:

  1. Test RMSE by architecture
  2. Best validation RMSE
  3. Model complexity (estimated parameters)
  4. Training loss trajectories
  5. Validation loss trajectories
  6. Performance metrics comparison

- **benchmark_results.json**: Numerical results for further analysis

**Benchmark with Custom Settings**:

```bash
python benchmark_all_architectures.py --epochs 100 --learning-rate 0.01
```

### 6. Data Analysis & RMSE Interpretation

Understand the data ranges and baseline metrics:

```bash
python analyze_data_ranges.py
```

Output shows:

- Feature ranges (SigC, SigH, EpsC, EpsH)
- Target (density) statistics
- Baseline naive prediction RMSE (201.44 kg/m³)

Interpret your model's RMSE:

```bash
python analyze_rmse.py
```

This displays RMSE as:

- % of density range
- % of standard deviation
- % of mean value
- Improvement over naive baseline

## Command-Line Arguments

### Architecture Selection

- `--architecture`: Choose model type: `mlp`, `cnn`, `cnn_multiscale`, or `gnn` (default: from config.py)

### MLP-Specific

- `--num-layers`: Number of hidden layers (default: 1)
- `--expansion-factor`: Hidden dimension multiplier (default: 8)
- `--use-swiglu`: Use SwiGLU activation (default: true)

### CNN-Specific

- `--cnn-num-layers`: Number of convolutional layers (default: 4)
- `--cnn-kernel-size`: Convolutional kernel size (default: 3)
- `--cnn-expansion-size`: Channel expansion (default: 8)
- `--cnn-use-batch-norm`: Enable batch normalization (default: true)
- `--cnn-use-residual`: Enable residual connections (default: false for CNN, true for Multi-Scale)

### GNN-Specific

- `--gnn-type`: Graph type: `gat` or `gcn` (default: gat)
- `--gnn-hidden-dim`: Hidden dimension for node embeddings (default: 64)
- `--gnn-num-layers`: Number of graph convolution layers (default: 3)
- `--dropout-rate`: Dropout probability (default: 0.2)

### Training

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--learning-rate`: Learning rate (default: 0.01)
- `--optimizer`: `adam` or `sgd` (default: adam)
- `--loss`: `mse` or `mae` (default: mse)

### Data

- `--data-dir`: Directory with CSV files (default: current directory)
- `--val-split`: Validation split ratio (default: 0.18)
- `--test-split`: Test split ratio (default: 0.1)
- `--no-normalize`: Disable input/output normalization (default: false)

### General

- `--device`: `cuda` or `cpu` (default: auto-detect)
- `--seed`: Random seed (default: 42)
- `--output-dir`: Results output directory (default: ./results\_{architecture})

## Configuration Files

All configuration is managed through `config.py` with dataclasses:

### ModelConfig

Defines the architecture and model hyperparameters:

```python
@dataclass
class ModelConfig:
    # Architecture selection
    architecture: str = "gnn"                    # "mlp", "cnn", "cnn_multiscale", "gnn"

    # Common
    input_dim: int = 4
    output_dim: int = 1
    device: str = "cuda"
    dtype: str = "float32"

    # MLP parameters
    num_layers: int = 1
    expansion_factor: float = 8
    use_swiglu: bool = True

    # CNN parameters
    cnn_num_layers: int = 4
    cnn_kernel_size: int = 3
    cnn_expansion_size: int = 8
    cnn_use_batch_norm: bool = True
    cnn_use_residual: bool = False

    # Multi-Scale CNN parameters
    cnn_multiscale_num_scales: int = 3
    cnn_multiscale_base_channels: int = 16
    cnn_multiscale_expansion_size: int = 8

    # GNN parameters
    gnn_type: str = "gat"                       # "gat" or "gcn"
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    dropout_rate: float = 0.2
```

### TrainingConfig

Defines training hyperparameters:

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    batch_size: int = 64
    num_epochs: int = 100
    validation_split: float = 0.18
    test_split: float = 0.1
    random_seed: int = 42
    loss_fn: str = "mse"                        # "mse" or "mae"
    optimizer: str = "adam"                     # "adam" or "sgd"
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    save_best_model: bool = True
```

## Understanding the Data

### Input Features

The model expects 4 LJ potential parameters:

- **SigC**: Sigma for carbon (Ångströms), range: 0.05 - 0.35
- **SigH**: Sigma for hydrogen (Ångströms), range: 0.045 - 0.35
- **EpsC**: Epsilon for carbon (kcal/mol), range: 0.15 - 1.15
- **EpsH**: Epsilon for hydrogen (kcal/mol), range: 0.01 - 0.15

### Output Target

- **Density**: Predicted chemical density (kg/m³), range: 0 - 939.21
- **Statistics**: Mean = 604.52, Std = 201.44
- **Baseline RMSE** (always predict mean): 201.44 kg/m³

### Dataset Format

CSV files with shape (5, N) where N is number of samples:

```
,Sample_1,Sample_2,...,Sample_N
SigC,value1,value2,...,valueN
SigH,value1,value2,...,valueN
EpsC,value1,value2,...,valueN
EpsH,value1,value2,...,valueN
Density,value1,value2,...,valueN
```

All four Dataset\_\*.csv files are automatically combined during training.

## Training Process

### Typical Workflow

1. **Setup Environment** (one-time)

   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1      # Windows
   pip install -r requirements.txt
   ```

2. **Choose Architecture** (edit config.py or use CLI flag)

   ```bash
   python train.py --architecture gnn --epochs 100
   ```

3. **Monitor Training**

   - Validation loss tracked during training
   - Best model saved automatically
   - Training history saved to results directory

4. **Evaluate Results**

   ```bash
   python analyze_rmse.py
   ```

5. **Compare Architectures** (optional)
   ```bash
   python benchmark_all_architectures.py --epochs 100
   ```

### Output Structure

After training, results are saved to `results_{architecture}/`:

```
results_gnn/
├── best_model.pt              # Best model weights
├── model_config.json          # Architecture configuration
├── training_config.json       # Training hyperparameters
├── normalization_stats.json   # Feature/target statistics
├── training_history.json      # Train/val loss curves
├── test_results.json          # Test RMSE and metrics
└── predictions.pt             # Test predictions & targets
```

## Performance & Optimization

### Architecture Characteristics

| Aspect         | MLP          | CNN            | Multi-Scale CNN      | GNN                  |
| -------------- | ------------ | -------------- | -------------------- | -------------------- |
| **Speed**      | ⚡ Very Fast | 🔶 Fast        | 🟡 Medium            | 🔴 Slower            |
| **Accuracy**   | ✓ Good       | ✓ Good         | ⭐ Better            | 🏆 Best              |
| **Parameters** | ~6K          | ~16K           | ~40K                 | ~25K                 |
| **Best For**   | Baselines    | Local patterns | Multi-scale features | Feature interactions |
| **GPU Memory** | Minimal      | Low            | Moderate             | Moderate             |

### Optimization Tips

**For Better Accuracy**:

- Use GNN or Multi-Scale CNN architecture
- Train for longer (100+ epochs)
- Reduce learning rate (0.005) for stability
- Ensemble multiple models

**For Speed**:

- Use MLP architecture
- Reduce batch size to 32
- Decrease number of layers
- Use CPU inference for edge devices

**For Memory Constraints**:

- Use MLP or single-scale CNN
- Reduce batch size
- Use int8 quantization
- Reduce hidden dimensions via config

## Troubleshooting

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution**: Reinstall PyTorch with correct CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

**Solutions**:

- Reduce `--batch-size` (e.g., 32 or 16)
- Use CPU: `--device cpu`
- Reduce architecture complexity
- Use smaller `--expansion-factor` (MLP only)

### Poor Performance / High RMSE

**Solutions**:

- Train longer: increase `--epochs`
- Try different architecture: `--architecture gnn`
- Adjust learning rate: `--learning-rate 0.005`
- Normalize data: remove `--no-normalize`
- Check data quality: `python analyze_data_ranges.py`

### Data Loading Issues

**Solutions**:

- Verify Dataset\_\*.csv files exist in project root
- Check file format (see Data Format section)
- Ensure no missing values in CSV files

## Performance Analysis

See `PERFORMANCE_ANALYSIS.md` for detailed interpretation of:

- RMSE in practical terms
- Quality assessment framework
- Deployment considerations
- Example predictions with error bands

## References

**Papers**:

- Shazeer, N., & Stern, M. (2020). "GLU Variants Improve Transformer"
- Veličković, P., et al. (2017). "Graph Attention Networks"
- Kipf, T., & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks"

**PyTorch Documentation**:

- https://pytorch.org/docs/stable/nn.html
- https://pytorch-geometric.readthedocs.io/

## Example Workflows

### Workflow 1: Quick MLP Training

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy pandas matplotlib scikit-learn

# Train for 50 epochs (quick)
python train.py --architecture mlp --epochs 50 --learning-rate 0.01

# View results
python analyze_rmse.py
```

### Workflow 2: GNN Training (Recommended)

```bash
# Edit config.py: set architecture = "gnn"
python train.py --epochs 100

# Analyze performance
python analyze_rmse.py

# View predictions
python -c "import torch; data = torch.load('results_gnn/predictions.pt'); print(f'Predictions shape: {data[0].shape}')"
```

### Workflow 3: Full Architecture Comparison

```bash
# Benchmark all 4 architectures (takes ~30 minutes total)
python benchmark_all_architectures.py --epochs 50

# View comparative plots
# benchmark_results.png shows all comparisons
# benchmark_results.json contains numerical results
```

### Workflow 4: Hyperparameter Tuning

```bash
# Compare different learning rates with GNN
for lr in 0.001 0.005 0.01 0.05; do
  python train.py --architecture gnn --learning-rate $lr --epochs 100
  python analyze_rmse.py
done
```

## Citation

If you use this model in your research, please cite:

```bibtex
@software{chemical_density_surrogate,
  title={Chemical Density Surrogate Model with Multiple Architectures},
  author={Projekt zur Datenanalyse},
  year={2025},
  url={https://github.com/Projekt-zur-Datenanalyse/density}
}
```

## License

[License information to be added]

## Contributing

We welcome contributions! Areas for improvement:

- Additional architectures (Transformers, ResNets)
- Uncertainty quantification
- Active learning for sampling
- Improved data visualization
- Deployment optimization

Please open a GitHub issue or pull request.
