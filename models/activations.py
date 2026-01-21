"""
Activation functions and layer building blocks.

This module provides activation functions for neural network architectures:
- SwiGLU: GLU-variant activation with swish gating
- Standard activations: ReLU, SiLU, LeakyReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function with projection.
    
    Reference: "GLU Variants Improve Transformer" (Shazeer et al., 2020)
    
    SwiGLU splits the input in half, applies SiLU (Swish) to one half as a gate,
    and multiplies it with the other half. Includes an output projection.
    
    Formula:
        x1, x2 = split(input)
        output = Linear(SiLU(x1) * x2)
    
    For use in MLP: Linear(in, hidden*2) -> SwiGLU(hidden*2, out)
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize SwiGLU layer.
        
        Args:
            input_dim: Input dimension (should be 2x the hidden dim)
            output_dim: Output dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim // 2
        
        # Output projection
        self.proj = nn.Linear(self.hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Split into gate and data paths
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply swish gate and multiply
        hidden = F.silu(x1) * x2
        
        # Project to output dimension
        return self.proj(hidden)


__all__ = ["SwiGLU"]
