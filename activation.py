"""Activation functions and layer utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Swish activation applied
        """
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU activation function.
    
    Reference: "GLU Variants Improve Transformer" (Shazeer et al., 2020)
    This is a gated linear unit using Swish activation:
    SwiGLU(x) = (x * W + b) ⊗ swish(x * V + c)
    where ⊗ is element-wise multiplication.
    
    For simplicity, this implementation applies SwiGLU-style gating 
    by expanding the hidden dimension by 2x internally.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize SwiGLU layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
        """
        super().__init__()
        # Project to 2x hidden_dim for the gate mechanism
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Project and split for gate
        projected = self.fc1(x)
        value, gate = projected.chunk(2, dim=-1)
        
        # Apply gating: value * swish(gate)
        gated = value * torch.sigmoid(gate) * gate
        
        # Project to output dimension
        out = self.fc2(gated)
        return out


class MLPBlock(nn.Module):
    """A single MLP block with configurable activation (SwiGLU or Swish).
    
    This module can be stacked to create deeper networks while maintaining
    residual connections when needed.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_swiglu: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Initialize MLP block.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            use_swiglu: If True, use SwiGLU; if False, use Swish
            dropout_rate: Dropout rate after activation (0.0 = no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            self.activation_block = SwiGLU(input_dim, hidden_dim, output_dim)
        else:
            # Simple MLP with Swish: input -> hidden -> output
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.swish = Swish()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.activation_block = None
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP block.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        if self.use_swiglu:
            out = self.activation_block(x)
        else:
            out = self.fc1(x)
            out = self.swish(out)
            out = self.fc2(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
