"""
Core package for Chemical Density Surrogate.

This package provides the fundamental building blocks:
- Configuration management
- Data loading and preprocessing
- Training utilities
- Common utility functions

Usage:
    from core import TrainingConfig, DataLoader, Trainer
"""

from .config import TrainingConfig, DEFAULT_TRAINING_CONFIG
from .data_loader import DataLoader, Dataset
from .trainer import Trainer
from .utils import set_seed, get_device, denormalize

__all__ = [
    # Configuration
    "TrainingConfig",
    "DEFAULT_TRAINING_CONFIG",
    # Data loading
    "DataLoader",
    "Dataset",
    # Training
    "Trainer",
    # Utilities
    "set_seed",
    "get_device",
    "denormalize",
]
