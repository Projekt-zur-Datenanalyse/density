"""Activation functions and layer utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SOTA SwiGLU activation function (Llama-style).
    
    Reference: "GLU Variants Improve Transformer" (Shazeer et al., 2020)
    
    The SOTA implementation uses:
    - Unbounded gate with SiLU activation (not hard sigmoid)
    - Separate projections for gate and data paths
    - Element-wise multiplication of gated and data values
    
    Formula:
        gate = SiLU(W_gate @ x)
        data = W_data @ x
        output = W_down @ (gate * data)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.0):
        """Initialize SOTA SwiGLU layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (expanded dimension)
            output_dim: Output feature dimension
            dropout_rate: Dropout rate after gating (0.0 = no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Path A: Data projection (no bias for efficiency)
        self.w_up = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Path B: Gate projection with SiLU (unbounded, no hard sigmoid)
        self.w_gate = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Output projection back to original dimension
        self.w_down = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Dropout after gating
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SOTA SwiGLU activation.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Path B: Apply SiLU to gate projection (unbounded)
        gate = F.silu(self.w_gate(x))
        
        # Path A: Data projection
        data = self.w_up(x)
        
        # Element-wise multiplication
        gated = gate * data
        
        # Apply dropout after non-linearity
        if self.dropout is not None:
            gated = self.dropout(gated)
        
        # Project back to output dimension
        output = self.w_down(gated)
        
        return output



class MLPBlock(nn.Module):
    """A single MLP block with configurable activation (SwiGLU or standard SiLU).
    
    This module can be stacked to create deeper networks. Supports both:
    - SOTA SwiGLU: Two separate paths for gate and data with element-wise multiplication
    - Standard SiLU: Simple linear -> SiLU -> linear MLP
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
            hidden_dim: Hidden/expanded dimension
            output_dim: Output feature dimension
            use_swiglu: If True, use SOTA SwiGLU; if False, use standard SiLU MLP
            dropout_rate: Dropout rate after activation (0.0 = no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_swiglu = use_swiglu
        self.dropout_rate = dropout_rate
        
        if use_swiglu:
            # SOTA SwiGLU implementation
            self.swiglu = SwiGLU(input_dim, hidden_dim, output_dim, dropout_rate)
        else:
            # Standard SiLU MLP: input -> hidden -> output
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP block.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        if self.use_swiglu:
            # Use SOTA SwiGLU
            return self.swiglu(x)
        else:
            # Standard MLP with SiLU
            x = self.fc1(x)
            x = F.leaky_relu(x) # F.silu(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.fc2(x)
            return x
