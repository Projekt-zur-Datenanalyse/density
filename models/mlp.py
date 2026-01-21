"""
Multi-Layer Perceptron (MLP) architecture for chemical density prediction.

This module provides a simple, configurable MLP model for the surrogate
modeling task with support for multiple activation functions.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Literal

from .activations import SwiGLU


# Supported activation types
ACTIVATION_TYPES = ["relu", "silu", "leakyrelu", "swiglu"]
ActivationType = Literal["relu", "silu", "leakyrelu", "swiglu"]


def get_activation(activation: str) -> nn.Module:
    """Get activation module by name.
    
    Args:
        activation: One of "relu", "silu", "leakyrelu"
        
    Returns:
        Activation module instance
    """
    activations = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "leakyrelu": nn.LeakyReLU,
    }
    
    if activation not in activations:
        raise ValueError(
            f"Unknown activation '{activation}'. "
            f"Must be one of: {list(activations.keys())}"
        )
    
    return activations[activation]()


class MLP(nn.Module):
    """Multi-Layer Perceptron surrogate model for chemical density prediction.
    
    A simple, configurable feedforward neural network that maps Lennard-Jones
    parameters to density predictions.
    
    Default architecture (4 -> 16 -> 32 -> 8 -> 1):
        Linear(4, 16) -> ReLU -> Linear(16, 32) -> ReLU -> 
        Linear(32, 8) -> ReLU -> Linear(8, 1)
    
    Supports multiple activation types:
        - relu: Standard ReLU activation
        - silu: Swish/SiLU activation  
        - leakyrelu: Leaky ReLU with negative slope
        - swiglu: SwiGLU (Linear -> split -> swish gate -> project)
    
    Args:
        input_dim: Number of input features (default: 4)
        output_dim: Number of outputs (default: 1)
        hidden_dims: List of hidden layer sizes (default: [16, 32, 8])
        activation: Activation type (default: "relu")
        dropout_rate: Dropout probability (default: 0.0)
    
    Example:
        >>> model = MLP(hidden_dims=[16, 32, 8], activation="relu")
        >>> x = torch.randn(32, 4)  # batch of 32, 4 features
        >>> y = model(x)  # shape: (32, 1)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        activation: ActivationType = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [16, 32, 8]
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        
        # Build the network
        self.network = self._build_network()
    
    def _build_network(self) -> nn.Sequential:
        """Build the sequential network."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            is_last = (i == len(dims) - 2)
            
            if self.activation_type == "swiglu" and not is_last:
                # SwiGLU: Linear(in, out*2) -> SwiGLU -> out
                layers.append(nn.Linear(in_dim, out_dim * 2))
                layers.append(SwiGLU(out_dim * 2, out_dim))
            else:
                # Standard: Linear -> Activation
                layers.append(nn.Linear(in_dim, out_dim))
                
                if not is_last:
                    layers.append(get_activation(self.activation_type))
            
            # Add dropout after activation (except last layer)
            if not is_last and self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Features: [SigC, SigH, EpsC, EpsH]
        
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        return (
            f"MLP(dims={dims}, activation={self.activation_type}, "
            f"dropout={self.dropout_rate}, params={self.get_num_parameters():,})"
        )


__all__ = ["MLP", "ACTIVATION_TYPES", "ActivationType", "get_activation"]
