"""
Configuration classes for training and experiments.

This module provides configuration dataclasses that define all
training parameters in a clean, validated way.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training experiments.
    
    This is a unified configuration class that works with all architectures.
    Architecture-specific model parameters should be passed directly to
    the model creation functions.
    
    Attributes:
        # Data settings
        data_path: Path to the dataset CSV file
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        batch_size: Batch size for training
        
        # Training settings
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength
        optimizer: Optimizer name ("adam" or "sgd")
        scheduler: LR scheduler ("none", "cosine", "onecycle")
        
        # Loss settings
        loss_fn: Loss function ("mse" or "mae")
        
        # Device and reproducibility
        device: Device to use ("cuda", "cpu", or "auto")
        seed: Random seed for reproducibility
        
        # Output settings
        output_dir: Directory to save results
        save_checkpoints: Whether to save model checkpoints
        
        # Display settings
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        show_progress_bar: Whether to show training progress bars
    """
    # Data settings
    data_path: str = "dataset.csv"
    validation_split: float = 0.15
    test_split: float = 0.10
    batch_size: int = 64
    num_workers: int = 0
    
    # Training settings
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    
    # Scheduler-specific settings
    cosine_eta_min: float = 1e-6
    onecycle_pct_start: float = 0.3
    
    # Loss settings
    loss_fn: str = "mse"
    
    # Device and reproducibility
    device: str = "auto"
    seed: int = 46
    
    # Output settings
    output_dir: str = "results"
    save_checkpoints: bool = True
    
    # Display settings
    verbose: int = 1
    show_progress_bar: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate splits
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 <= self.test_split < 1:
            raise ValueError("test_split must be between 0 and 1")
        if self.validation_split + self.test_split >= 1:
            raise ValueError("validation_split + test_split must be < 1")
        
        # Validate optimizer
        if self.optimizer not in ["adam", "sgd"]:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        
        # Validate scheduler
        if self.scheduler not in ["none", "cosine", "onecycle"]:
            raise ValueError("scheduler must be 'none', 'cosine', or 'onecycle'")
        
        # Validate loss function
        if self.loss_fn not in ["mse", "mae"]:
            raise ValueError("loss_fn must be 'mse' or 'mae'")
        
        # Validate device
        if self.device not in ["auto", "cuda", "cpu"]:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data_path": self.data_path,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "cosine_eta_min": self.cosine_eta_min,
            "onecycle_pct_start": self.onecycle_pct_start,
            "loss_fn": self.loss_fn,
            "device": self.device,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "save_checkpoints": self.save_checkpoints,
            "verbose": self.verbose,
            "show_progress_bar": self.show_progress_bar,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


# Default configuration instance
DEFAULT_TRAINING_CONFIG = TrainingConfig()


__all__ = ["TrainingConfig", "DEFAULT_TRAINING_CONFIG"]
