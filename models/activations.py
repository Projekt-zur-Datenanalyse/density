"""
Activation functions and layer building blocks.

This module provides:
- SwiGLU: GLU-variant activation function
- MLPBlock: Configurable MLP block with optional SwiGLU/SiLU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function (Llama-style) without down-projection.
    
    Reference: "GLU Variants Improve Transformer" (Shazeer et al., 2020)
    
    This is a GLU-variant that operates on the hidden dimension without down-projecting back.
    It's designed to be used as an activation function within a layer, maintaining the hidden dim.
    
    Formula:
        gate = SiLU(W_gate @ x)  # Up-projection with SiLU
        data = W_data @ x         # Up-projection for data path
        output = gate * data      # Element-wise multiplication (no down-projection)
    
    Note: When using SwiGLU, pre-normalization is MANDATORY because there's no non-linearity
    after the element-wise multiplication. The normalization must happen before this layer.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.0):
        """Initialize SwiGLU layer (no down-projection).
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (same as hidden/expanded dimension)
            dropout_rate: Dropout rate after gating (0.0 = no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Path A: Data projection (up to output_dim)
        self.w_up = nn.Linear(input_dim, output_dim, bias=True)
        
        # Path B: Gate projection with SiLU (up to output_dim, unbounded)
        self.w_gate = nn.Linear(input_dim, output_dim, bias=True)
        
        # Dropout after gating
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Path B: Apply SiLU to gate projection (unbounded)
        gate = F.silu(self.w_gate(x))
        
        # Path A: Data projection (no activation)
        data = self.w_up(x)
        
        # Element-wise multiplication (no projection after this!)
        gated = gate * data
        
        # Apply dropout after non-linearity
        if self.dropout is not None:
            gated = self.dropout(gated)
        
        return gated


class MLPBlock(nn.Module):
    """A single MLP block with proper sequential dimension flow.
    
    This module implements a proper feed-forward block that:
    - Takes input_dim and projects to hidden_dim (expansion/contraction as needed)
    - Applies activation (SwiGLU or standard SiLU)
    - Returns output of size output_dim
    - Supports optional pre-normalization (MANDATORY for SwiGLU)
    - Supports optional residual connections (when input_dim == output_dim)
    
    Key design:
    - For standard FFN: input -> fc1(hidden) -> activation -> fc2(output)
    - For SwiGLU: input -> [gate(hidden), data(hidden)] -> gate*data -> fc2(output)
    - Pre-normalization applied BEFORE the expansion (required for SwiGLU)
    - Residuals only when input_dim == output_dim
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_swiglu: bool = True,
        dropout_rate: float = 0.0,
        use_prenorm: bool = False,
        norm_type: str = "layer",  # "layer" or "batch"
        residual: bool = True,
    ):
        """Initialize MLP block.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Intermediate/hidden dimension (expansion point)
            output_dim: Output feature dimension
            use_swiglu: If True, use SwiGLU; if False, use standard SiLU MLP
            dropout_rate: Dropout rate (0.0 = no dropout)
            use_prenorm: If True, apply pre-normalization before block
            norm_type: "layer" for LayerNorm, "batch" for BatchNorm
            residual: If True and input_dim == output_dim, add residual connection
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_swiglu = use_swiglu
        self.dropout_rate = dropout_rate
        self.use_prenorm = use_prenorm
        self.residual = residual and (input_dim == output_dim)
        
        # Pre-normalization (MANDATORY for SwiGLU, optional for standard)
        if use_prenorm:
            if norm_type == "layer":
                self.norm = nn.LayerNorm(input_dim)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm1d(input_dim)
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
        else:
            self.norm = None
        
        if use_swiglu:
            # SwiGLU: Up-project to hidden_dim, apply gate and data paths, multiply
            self.swiglu = SwiGLU(input_dim, hidden_dim, dropout_rate)
            # Down-project from hidden_dim to output_dim
            self.fc_down = nn.Linear(hidden_dim, output_dim, bias=True)
        else:
            # Standard MLP: input -> hidden -> output
            # Up-project to hidden dimension
            self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
            # Down-project to output dimension
            self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
            # Activation and dropout
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP block with optional residuals and pre-normalization.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Store input for potential residual connection
        residual = x
        
        # Pre-normalization (if enabled)
        if self.norm is not None:
            x = self.norm(x)
        
        if self.use_swiglu:
            # Apply SwiGLU (produces hidden_dim output)
            x = self.swiglu(x)  # (..., hidden_dim)
            # Project down to output_dim
            x = self.fc_down(x)  # (..., output_dim)
        else:
            # Standard MLP with SiLU
            x = self.fc1(x)  # (..., hidden_dim)
            x = F.silu(x)  # Apply SiLU activation
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.fc2(x)  # (..., output_dim)
        
        # Add residual connection if applicable
        if self.residual:
            x = x + residual
        
        return x


__all__ = ["SwiGLU", "MLPBlock"]
