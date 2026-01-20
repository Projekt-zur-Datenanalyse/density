"""
Common utility functions.

This module provides utility functions used across the codebase:
- Seed management for reproducibility
- Device detection
- Denormalization helpers
"""

import numpy as np
import torch
from typing import Union, Optional


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA (if available)
    - CUDNN deterministic mode
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification:
            - "auto": Automatically select CUDA if available, else CPU
            - "cuda": Force CUDA (raises error if unavailable)
            - "cpu": Force CPU
    
    Returns:
        torch.device object
        
    Raises:
        RuntimeError: If CUDA is requested but not available
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device}. Use 'auto', 'cuda', or 'cpu'.")


def denormalize(
    values: Union[np.ndarray, torch.Tensor, float],
    mean: float,
    std: float,
) -> Union[np.ndarray, torch.Tensor, float]:
    """Denormalize values using mean and standard deviation.
    
    Formula: original = normalized * std + mean
    
    Args:
        values: Normalized values to denormalize
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Denormalized values in same type as input
    """
    return values * std + mean


def normalize(
    values: Union[np.ndarray, torch.Tensor, float],
    mean: float,
    std: float,
) -> Union[np.ndarray, torch.Tensor, float]:
    """Normalize values using mean and standard deviation.
    
    Formula: normalized = (original - mean) / std
    
    Args:
        values: Values to normalize
        mean: Mean for normalization
        std: Standard deviation for normalization
    
    Returns:
        Normalized values in same type as input
    """
    return (values - mean) / std


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s" or "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model or any model with get_num_parameters method
        
    Returns:
        Number of trainable parameters
    """
    if hasattr(model, 'get_num_parameters'):
        return model.get_num_parameters()
    elif hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return 0


__all__ = [
    "set_seed",
    "get_device", 
    "denormalize",
    "normalize",
    "format_time",
    "count_parameters",
]
