"""
Multi-Layer Perceptron (MLP) architecture for chemical density prediction.

This module provides the MLP model which is the primary neural network
architecture for the surrogate modeling task.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .activations import MLPBlock


class MLP(nn.Module):
    """Multi-Layer Perceptron surrogate model for chemical density prediction.
    
    This model takes 4 input features (SigC, SigH, EpsC, EpsH) and predicts
    the density of a chemical structure. It uses a configurable multi-layer
    architecture with optional SwiGLU or Swish activation functions.
    
    Architecture:
    - Input layer: 4 features (Lennard-Jones parameters)
    - Hidden layers: Configurable via hidden_dims list
    - Output layer: 1 value (density prediction)
    
    Features:
    - Configurable hidden layer dimensions
    - Optional SwiGLU or Swish activation
    - Residual connections for deeper networks
    - Pre-normalization support
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        use_swiglu: bool = False,
        dropout_rate: float = 0.1,
        use_prenorm: bool = False,
        norm_type: str = "layer",
        expansion_factor: float = 4.0,
    ):
        """Initialize the MLP model.
        
        Args:
            input_dim: Number of input features (default: 4 for LJ parameters)
            output_dim: Number of output values (default: 1 for density)
            hidden_dims: List of hidden layer dimensions. If None, uses expansion_factor.
            use_swiglu: Use SwiGLU activation (True) vs standard SiLU (False)
            dropout_rate: Dropout rate for regularization
            use_prenorm: Use pre-normalization before each block (MANDATORY for SwiGLU)
            norm_type: "layer" for LayerNorm, "batch" for BatchNorm
            expansion_factor: Hidden dim multiplier if hidden_dims is None
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_swiglu = use_swiglu
        self.dropout_rate = dropout_rate
        
        # Determine hidden layer structure
        if hidden_dims is not None and len(hidden_dims) > 0:
            self.hidden_dims = hidden_dims
        else:
            # Default: single hidden layer with expansion
            hidden_dim = int(input_dim * expansion_factor)
            self.hidden_dims = [hidden_dim]
        
        # For SwiGLU, pre-normalization is MANDATORY
        if use_swiglu and not use_prenorm:
            use_prenorm = True
        
        # Build network
        self.layers = self._build_network(use_prenorm, norm_type)
    
    def _build_network(self, use_prenorm: bool, norm_type: str) -> nn.ModuleList:
        """Build the network layers.
        
        Args:
            use_prenorm: Whether to use pre-normalization
            norm_type: Type of normalization to use
            
        Returns:
            ModuleList of network layers
        """
        layers = nn.ModuleList()
        
        # Build layer sequence: input -> hidden1 -> hidden2 -> ... -> output
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            is_last = (i == len(dims) - 2)
            
            # Calculate hidden dimension for this layer's internal expansion
            # For intermediate layers, use the output dim as hidden
            # For the last layer, keep it simple
            if is_last:
                hidden = max(in_dim, out_dim)
            else:
                hidden = int(out_dim * 2)  # Expand internally
            
            # Determine if residual is possible
            can_residual = (in_dim == out_dim) and not is_last
            
            layers.append(
                MLPBlock(
                    input_dim=in_dim,
                    hidden_dim=hidden,
                    output_dim=out_dim,
                    use_swiglu=self.use_swiglu if not is_last else False,
                    dropout_rate=self.dropout_rate if not is_last else 0.0,
                    use_prenorm=use_prenorm if not is_last else False,
                    norm_type=norm_type,
                    residual=can_residual,
                )
            )
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Features: [SigC, SigH, EpsC, EpsH]
        
        Returns:
            Density predictions of shape (batch_size, output_dim)
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, "
                f"got {x.shape[-1]}"
            )
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        layer_str = " -> ".join(str(d) for d in dims)
        
        info = [
            "=" * 60,
            "MLP Surrogate Model",
            "=" * 60,
            f"Architecture: {layer_str}",
            f"Activation: {'SwiGLU' if self.use_swiglu else 'SiLU'}",
            f"Dropout rate: {self.dropout_rate}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(info)


__all__ = ["MLP"]
