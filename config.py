"""Configuration for the Chemical Density Surrogate Model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the surrogate model architecture.
    
    Supports multiple architectures:
    - 'mlp': Multi-Layer Perceptron (default)
    - 'cnn': Convolutional Neural Network
    - 'cnn_multiscale': Multi-Scale CNN (Inception-style)
    - 'gnn': Graph Neural Network (requires torch_geometric)
    """
    
    # Architecture selection
    architecture: str = "mlp"  # "mlp", "cnn", "cnn_multiscale", or "gnn"
    
    # Input/Output dimensions
    input_dim: int = 4  # SigC, SigH, EpsC, EpsH
    output_dim: int = 1  # Density prediction
    
    # MLP-specific configuration
    num_layers: int = 2  # Number of hidden layers (deep+narrow is best: 4 layers x 16 units)
    expansion_factor: float = 4  # Multiplier for hidden dimension (4 * 4 inputs = 16 hidden)
    use_swiglu: bool = False  # Use SwiGLU (False) vs Swish (True) - simpler for regression
    
    # CNN-specific configuration
    cnn_expansion_size: int = 8  # Expand 4 features to 8x8 spatial grid
    cnn_num_layers: int = 4  # Number of convolutional layers
    cnn_kernel_size: int = 3  # Convolutional kernel size
    cnn_use_batch_norm: bool = True  # Use batch normalization
    cnn_use_residual: bool = False  # Use residual connections
    
    # Multi-Scale CNN configuration
    cnn_multiscale_expansion_size: int = 16  # Spatial expansion size
    cnn_multiscale_num_scales: int = 3  # Number of parallel branches
    cnn_multiscale_base_channels: int = 16  # Base channels per branch
    
    # GNN-specific configuration
    gnn_hidden_dim: int = 64  # Hidden dimension for GNN embeddings
    gnn_num_layers: int = 3  # Number of GNN layers
    gnn_type: str = "gat"  # "gcn", "gat", or "graphconv"
    
    # Common configuration
    dropout_rate: float = 0.2  # Dropout rate for regularization
    
    # Training configuration
    device: str = "cuda" if hasattr(__import__('torch'), 'cuda') else "cpu"
    dtype: str = "float32"  # "float32" or "float64"
    
    def get_hidden_dim(self) -> int:
        """Calculate the hidden dimension size (MLP only)."""
        return int(self.input_dim * self.expansion_factor)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate architecture
        valid_archs = ["mlp", "cnn", "cnn_multiscale", "gnn"]
        if self.architecture not in valid_archs:
            raise ValueError(f"architecture must be one of {valid_archs}, got {self.architecture}")
        
        # MLP validation
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        if self.input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        
        # CNN validation
        if self.cnn_expansion_size < 1:
            raise ValueError("cnn_expansion_size must be at least 1")
        if self.cnn_num_layers < 1:
            raise ValueError("cnn_num_layers must be at least 1")
        if self.cnn_kernel_size < 1 or self.cnn_kernel_size % 2 == 0:
            raise ValueError("cnn_kernel_size must be positive and odd")
        
        # Multi-Scale CNN validation
        if self.cnn_multiscale_num_scales < 1:
            raise ValueError("cnn_multiscale_num_scales must be at least 1")
        
        # GNN validation
        if self.gnn_hidden_dim < 1:
            raise ValueError("gnn_hidden_dim must be at least 1")
        if self.gnn_num_layers < 1:
            raise ValueError("gnn_num_layers must be at least 1")
        if self.gnn_type not in ["gcn", "gat", "graphconv"]:
            raise ValueError("gnn_type must be 'gcn', 'gat', or 'graphconv'")


@dataclass
class TrainingConfig:
    """Configuration for training the surrogate model."""
    
    learning_rate: float = 0.01  # Learning rate for Adam optimizer (tested: 0.01 is optimal)
    batch_size: int = 64  # Batch size for training
    num_epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 46
    
    # Loss and optimization
    loss_fn: str = "mse"  # "mse" or "mae"
    optimizer: str = "adam"  # "adam" or "sgd"
    weight_decay: float = 1e-5  # L2 regularization
    dropout_rate: float = 0.2  # Dropout rate (0.1-0.3 typical)
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "none", "onecycle", or "cosine"
    onecycle_pct_start: float = 0.3  # Percentage of cycle spent increasing LR
    onecycle_anneal_strategy: str = "cos"  # "cos" or "linear"
    cosine_t_max: int = 100  # Max iterations for cosine annealing (default to num_epochs)
    cosine_eta_min: float = 1e-3  # Minimum learning rate for cosine annealing
    
    # Data normalization
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10  # Log every N batches
    show_progress_bar: bool = True  # Show progress bar per epoch
