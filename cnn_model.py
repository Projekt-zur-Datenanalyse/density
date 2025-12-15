"""
Convolutional Neural Network (CNN) architecture for chemical density prediction.
Expands the tiny 4-feature input and applies convolutional operations
to learn hierarchical feature representations.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvolutionalSurrogate(nn.Module):
    """CNN-based surrogate model for chemical density prediction.
    
    Strategy: Expand 4 features into a spatial representation, then apply
    1D/2D convolutions to learn complex feature interactions at multiple scales.
    
    Approach:
    1. Expand 4 features into a 2D grid (e.g., 2x2, 4x4, etc.)
    2. Apply multiple convolutional layers with different kernel sizes
    3. Use residual connections and batch normalization
    4. Global pooling to aggregate spatial information
    5. Fully connected output layers
    """
    
    def __init__(
        self,
        num_input_features: int = 4,
        expansion_size: int = 8,  # Expand 4 features to 8x8 grid
        num_conv_layers: int = 4,
        conv_channels: List[int] = None,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        """Initialize CNN surrogate model.
        
        Args:
            num_input_features: Number of input features (4 for SigC, SigH, EpsC, EpsH)
            expansion_size: Size to expand features to (e.g., 8 means 8x8 grid = 64 values)
            num_conv_layers: Number of convolutional layers
            conv_channels: List of channel sizes for each conv layer
            kernel_size: Kernel size for convolutions
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            dropout_rate: Dropout rate for regularization
            output_dim: Output dimension (1 for density)
        """
        super().__init__()
        
        self.num_input_features = num_input_features
        self.expansion_size = expansion_size
        self.num_conv_layers = num_conv_layers
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        # Default channel progression
        if conv_channels is None:
            conv_channels = [16, 32, 64, 128, 256][:num_conv_layers]
            if len(conv_channels) < num_conv_layers:
                # Extend if needed
                last_channels = conv_channels[-1] if conv_channels else 16
                conv_channels.extend([last_channels] * (num_conv_layers - len(conv_channels)))
        
        self.conv_channels = conv_channels
        
        # Feature expansion layer
        # Expand 4 features to a spatial grid
        expanded_size = expansion_size * expansion_size
        self.expand = nn.Sequential(
            nn.Linear(num_input_features, expanded_size),
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
            # Conv layer with padding to maintain spatial dimensions
            padding = kernel_size // 2
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batch_norm,  # No bias if using batch norm
                )
            )
            
            # Batch norm (optional)
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            self.activation_layers.append(nn.ReLU())
            
            # Dropout
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
            x: Input tensor of shape (batch_size, 4)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Expand features to spatial grid
        x_expanded = self.expand(x)  # (batch_size, expansion_size^2)
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
        x_pooled = self.global_pool(x_conv)  # (batch_size, channels, 1, 1)
        x_pooled = x_pooled.view(batch_size, -1)  # (batch_size, channels)
        
        # Output projection
        output = self.output_head(x_pooled)  # (batch_size, 1)
        
        return output
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture."""
        info_lines = [
            "=" * 60,
            "Convolutional Neural Network (CNN) Surrogate Model",
            "=" * 60,
            f"Input features: {self.num_input_features}",
            f"Expansion size: {self.expansion_size}x{self.expansion_size} = {self.expansion_size**2}",
            f"Number of conv layers: {self.num_conv_layers}",
            f"Channel progression: {self.conv_channels}",
            f"Batch normalization: {self.use_batch_norm}",
            f"Residual connections: {self.use_residual}",
            f"Dropout rate: {self.dropout_rate}",
            f"Output dimension: {self.output_dim}",
            f"Total parameters: {sum(p.numel() for p in self.parameters()):,}",
            "=" * 60,
        ]
        return "\n".join(info_lines)


class MultiScaleConvolutionalSurrogate(nn.Module):
    """Advanced CNN with multi-scale convolutions.
    
    Uses multiple parallel convolutional branches with different kernel sizes
    to capture features at different scales, similar to Inception modules.
    """
    
    def __init__(
        self,
        num_input_features: int = 4,
        expansion_size: int = 8,
        num_scales: int = 3,
        base_channels: int = 16,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        """Initialize multi-scale CNN.
        
        Args:
            num_input_features: Number of input features (4)
            expansion_size: Size of expanded grid (e.g., 8 = 8x8)
            num_scales: Number of parallel convolutional branches
            base_channels: Number of channels for each scale branch
            dropout_rate: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()
        
        expanded_size = expansion_size * expansion_size
        
        # Feature expansion
        self.expand = nn.Sequential(
            nn.Linear(num_input_features, expanded_size),
            nn.BatchNorm1d(expanded_size),
            nn.SiLU(),
        )
        
        # Multi-scale convolutional branches
        self.branches = nn.ModuleList()
        self.expansion_size = expansion_size
        
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
            x: Input tensor of shape (batch_size, 4)
        
        Returns:
            Output tensor of shape (batch_size, 1)
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
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture."""
        info_lines = [
            "=" * 60,
            "Multi-Scale CNN Surrogate Model (Inception-style)",
            "=" * 60,
            f"Input features: 4",
            f"Expansion size: {self.expansion_size}x{self.expansion_size}",
            f"Number of parallel scales: {len(self.branches)}",
            f"Total parameters: {sum(p.numel() for p in self.parameters()):,}",
            "=" * 60,
        ]
        return "\n".join(info_lines)
