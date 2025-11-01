"""Core surrogate model architecture for chemical density prediction."""

import torch
import torch.nn as nn
from typing import Optional, List

from config import ModelConfig
from activation import MLPBlock, Swish


class ChemicalDensitySurrogate(nn.Module):
    """Surrogate model for predicting chemical density from molecular features.
    
    This model takes 4 input features (SigC, SigH, EpsC, EpsH) and predicts
    the density of a chemical structure. It uses a configurable multi-layer
    MLP with optional SwiGLU or Swish activation functions.
    
    Architecture:
    - Input layer: 4 features
    - Hidden layers: num_layers blocks with residual connections (when num_layers > 1)
    - Output layer: 1 value (density)
    
    Features:
    - Configurable expansion factor for hidden dimensions
    - Optional SwiGLU or Swish activation
    - Residual connections for deeper networks
    - Modular and extensible design
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize the surrogate model.
        
        Args:
            config: ModelConfig instance specifying architecture parameters
        """
        super().__init__()
        self.config = config
        self._validate_config()
        
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.hidden_dim = config.get_hidden_dim()
        self.num_layers = config.num_layers
        self.use_swiglu = config.use_swiglu
        
        # Build the network
        self.layers = self._build_network()
    
    def _validate_config(self) -> None:
        """Validate the model configuration."""
        # Allow flexible input dimensions for feature engineering experiments
        if self.config.input_dim < 1:
            raise ValueError("Input dimension must be at least 1")
        if self.config.output_dim != 1:
            raise ValueError("Chemical density model should output 1 value (density)")
    
    def _build_network(self) -> nn.ModuleList:
        """Build the network layers.
        
        Returns:
            List of layers in the network
        """
        layers = nn.ModuleList()
        
        # Get dropout rate from config if available
        dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
        
        if self.num_layers == 1:
            # Single hidden layer: input -> hidden -> output
            layers.append(
                MLPBlock(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    use_swiglu=self.use_swiglu,
                    dropout_rate=dropout_rate,
                )
            )
        else:
            # First layer: input -> hidden
            layers.append(
                MLPBlock(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    use_swiglu=self.use_swiglu,
                    dropout_rate=dropout_rate,
                )
            )
            
            # Intermediate layers: hidden -> hidden (with residual connections)
            for _ in range(self.num_layers - 2):
                layers.append(
                    MLPBlock(
                        input_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                    )
                )
            
            # Last layer: hidden -> output
            layers.append(
                MLPBlock(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    use_swiglu=self.use_swiglu,
                    dropout_rate=dropout_rate,
                )
            )
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4)
               Features: [SigC, SigH, EpsC, EpsH]
        
        Returns:
            Density predictions of shape (batch_size, 1)
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, "
                f"got {x.shape[-1]}"
            )
        
        # Single layer case
        if self.num_layers == 1:
            return self.layers[0](x)
        
        # Multi-layer case with residual connections
        out = self.layers[0](x)  # First layer projects from input to hidden
        
        # Apply intermediate layers with residual connections
        for layer in self.layers[1:-1]:
            out = out + layer(out)  # Add residual connection
        
        # Last layer projects to output
        out = self.layers[-1](out)
        
        return out
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture.
        
        Returns:
            String describing the model configuration
        """
        info_lines = [
            "=" * 60,
            "Chemical Density Surrogate Model",
            "=" * 60,
            f"Input features: {self.input_dim} (SigC, SigH, EpsC, EpsH)",
            f"Output dimension: {self.output_dim} (Density)",
            f"Hidden dimension: {self.hidden_dim}",
            f"Expansion factor: {self.config.expansion_factor}x",
            f"Number of layers: {self.num_layers}",
            f"Activation: {'SwiGLU' if self.use_swiglu else 'Swish'}",
            f"Residual connections: {'Yes' if self.num_layers > 1 else 'No'}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(info_lines)
