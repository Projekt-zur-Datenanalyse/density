"""
Convolutional Neural Network (CNN) architectures for chemical density prediction.

This module provides CNN-based models that expand the small input feature space
into a spatial representation for convolutional processing.

Models:
- CNN: Standard convolutional network with residual connections
- MultiScaleCNN: Inception-style multi-scale convolutional network
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CNN(nn.Module):
    """CNN-based surrogate model for chemical density prediction.
    
    Strategy: Expand 4 features into a spatial representation, then apply
    1D/2D convolutions to learn complex feature interactions at multiple scales.
    
    Approach:
    1. Expand 4 features into a 2D grid (e.g., 8x8 = 64 values)
    2. Apply multiple convolutional layers with same kernel size
    3. Use residual connections and optional batch normalization
    4. Global pooling to aggregate spatial information
    5. Fully connected output layers
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        expansion_size: int = 8,
        num_layers: int = 4,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        use_residual: bool = True,
        dropout_rate: float = 0.2,
    ):
        """Initialize CNN surrogate model.
        
        Args:
            input_dim: Number of input features (default: 4)
            output_dim: Output dimension (default: 1 for density)
            expansion_size: Size to expand features to (e.g., 8 means 8x8 grid)
            num_layers: Number of convolutional layers
            conv_channels: List of channel sizes for each conv layer
            kernel_size: Kernel size for convolutions (must be odd)
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expansion_size = expansion_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
        # Default channel progression
        if conv_channels is None:
            conv_channels = [16, 32, 64, 128, 256][:num_layers]
            if len(conv_channels) < num_layers:
                last_channels = conv_channels[-1] if conv_channels else 16
                conv_channels.extend([last_channels] * (num_layers - len(conv_channels)))
        
        self.conv_channels = conv_channels
        
        # Feature expansion layer
        expanded_size = expansion_size * expansion_size
        self.expand = nn.Sequential(
            nn.Linear(input_dim, expanded_size),
            nn.BatchNorm1d(expanded_size) if use_batch_norm else nn.Identity(),
            nn.SiLU(),
        )
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() if use_batch_norm else None
        self.activation_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = 1
        for i, out_channels in enumerate(conv_channels):
            padding = kernel_size // 2
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batch_norm,
                )
            )
            
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm2d(out_channels))
            
            self.activation_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Expand features to spatial grid
        x_expanded = self.expand(x)
        x_spatial = x_expanded.view(batch_size, 1, self.expansion_size, self.expansion_size)
        
        # Apply convolutional layers with residual connections
        x_conv = x_spatial
        for i, conv_layer in enumerate(self.conv_layers):
            x_conv_out = conv_layer(x_conv)
            
            if self.use_batch_norm:
                x_conv_out = self.batch_norm_layers[i](x_conv_out)
            
            x_conv_out = self.activation_layers[i](x_conv_out)
            x_conv_out = self.dropout_layers[i](x_conv_out)
            
            # Residual connection (skip connection if compatible shapes)
            if self.use_residual and x_conv.shape[1] == x_conv_out.shape[1]:
                x_conv_out = x_conv_out + x_conv
            
            x_conv = x_conv_out
        
        # Global average pooling
        x_pooled = self.global_pool(x_conv)
        x_pooled = x_pooled.view(batch_size, -1)
        
        # Output projection
        output = self.output_head(x_pooled)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture."""
        info = [
            "=" * 60,
            "Convolutional Neural Network (CNN) Surrogate Model",
            "=" * 60,
            f"Input features: {self.input_dim}",
            f"Expansion size: {self.expansion_size}x{self.expansion_size}",
            f"Number of conv layers: {self.num_layers}",
            f"Channel progression: {self.conv_channels}",
            f"Batch normalization: {self.use_batch_norm}",
            f"Residual connections: {self.use_residual}",
            f"Dropout rate: {self.dropout_rate}",
            f"Output dimension: {self.output_dim}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(info)


class MultiScaleCNN(nn.Module):
    """Advanced CNN with multi-scale convolutions (Inception-style).
    
    Uses multiple parallel convolutional branches with different kernel sizes
    to capture features at different scales simultaneously.
    
    Architecture:
    1. Expand features to spatial grid
    2. Apply parallel branches with kernel sizes 3, 5, 7, ...
    3. Each branch: Conv -> BN -> SiLU -> Conv -> BN -> SiLU -> Pool
    4. Concatenate branch outputs
    5. Fully connected output head
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        expansion_size: int = 16,
        num_scales: int = 3,
        base_channels: int = 16,
        dropout_rate: float = 0.2,
    ):
        """Initialize multi-scale CNN.
        
        Args:
            input_dim: Number of input features (default: 4)
            output_dim: Output dimension (default: 1)
            expansion_size: Size of expanded grid (e.g., 16 = 16x16)
            num_scales: Number of parallel convolutional branches
            base_channels: Number of channels for each scale branch
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expansion_size = expansion_size
        self.num_scales = num_scales
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        
        expanded_size = expansion_size * expansion_size
        
        # Feature expansion
        self.expand = nn.Sequential(
            nn.Linear(input_dim, expanded_size),
            nn.BatchNorm1d(expanded_size),
            nn.SiLU(),
        )
        
        # Multi-scale convolutional branches
        self.branches = nn.ModuleList()
        
        for scale in range(num_scales):
            kernel_size = 3 + 2 * scale  # 3, 5, 7, ...
            padding = kernel_size // 2
            
            branch = nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(base_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.branches.append(branch)
        
        # Concatenation and output
        total_features = base_channels * num_scales
        self.output_head = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale CNN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Expand to spatial grid
        x_expanded = self.expand(x)
        x_spatial = x_expanded.view(batch_size, 1, self.expansion_size, self.expansion_size)
        
        # Apply parallel branches
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x_spatial)
            branch_out = branch_out.view(batch_size, -1)
            branch_outputs.append(branch_out)
        
        # Concatenate branch outputs
        x_concat = torch.cat(branch_outputs, dim=1)
        
        # Output projection
        output = self.output_head(x_concat)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture."""
        kernel_sizes = [3 + 2 * i for i in range(self.num_scales)]
        info = [
            "=" * 60,
            "Multi-Scale CNN Surrogate Model (Inception-style)",
            "=" * 60,
            f"Input features: {self.input_dim}",
            f"Expansion size: {self.expansion_size}x{self.expansion_size}",
            f"Number of parallel scales: {self.num_scales}",
            f"Kernel sizes: {kernel_sizes}",
            f"Base channels per branch: {self.base_channels}",
            f"Dropout rate: {self.dropout_rate}",
            f"Output dimension: {self.output_dim}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(info)


__all__ = ["CNN", "MultiScaleCNN"]
