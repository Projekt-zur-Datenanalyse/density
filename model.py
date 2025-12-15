"""Core surrogate model architecture for chemical density prediction."""

import torch
import torch.nn as nn
from typing import Optional, List

from config import ModelConfig
from activation import MLPBlock


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
        
        # Determine layer structure: use custom hidden_layer_dims if provided
        if config.hidden_layer_dims is not None:
            self.hidden_layer_dims = config.hidden_layer_dims
            self.num_layers = len(config.hidden_layer_dims)  # Update num_layers based on dims
            self.use_expansion_factor = False
        else:
            # Use traditional expansion_factor approach
            self.hidden_layer_dims = None
            self.num_layers = config.num_layers
            self.use_expansion_factor = True
            self.hidden_dim = config.get_hidden_dim()
        
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
        """Build the network layers with proper sequential dimension flow.
        
        Constructs layers that properly flow through dimensions:
        - Each layer takes output of previous layer as input
        - Expansion happens at each layer (via hidden_dim)
        - Residuals added only when input_dim == output_dim
        - Pre-normalization applied before each layer (MANDATORY for SwiGLU)
        
        Returns:
            List of layers in the network
        """
        layers = nn.ModuleList()
        
        # Get dropout rate and normalization settings from config
        dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
        # For SwiGLU, pre-normalization is MANDATORY
        use_prenorm = self.use_swiglu or getattr(self.config, 'use_prenorm', False)
        norm_type = getattr(self.config, 'norm_type', 'layer')
        
        # Case 1: Custom hidden layer dimensions provided
        if self.hidden_layer_dims is not None:
            if len(self.hidden_layer_dims) == 0:
                # Empty array: direct projection from input to output (no hidden layers)
                layers.append(
                    MLPBlock(
                        input_dim=self.input_dim,
                        hidden_dim=max(self.input_dim, self.output_dim),
                        output_dim=self.output_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                        use_prenorm=use_prenorm,
                        norm_type=norm_type,
                        residual=False,
                    )
                )
            else:
                # Build sequence of layers with proper dimension flow
                # hidden_layer_dims defines the output dimension of each layer
                # The hidden expansion happens within each MLPBlock
                
                current_dim = self.input_dim
                
                for i, next_dim in enumerate(self.hidden_layer_dims):
                    # Calculate expansion factor for this layer
                    # Use expansion_factor if available, otherwise keep hidden roughly 2x
                    expansion_factor = getattr(self.config, 'expansion_factor', 2.0)
                    hidden_dim = max(current_dim, next_dim)
                    hidden_dim = int(hidden_dim * expansion_factor)
                    
                    layers.append(
                        MLPBlock(
                            input_dim=current_dim,
                            hidden_dim=hidden_dim,
                            output_dim=next_dim,
                            use_swiglu=self.use_swiglu,
                            dropout_rate=dropout_rate,
                            use_prenorm=use_prenorm,
                            norm_type=norm_type,
                            residual=True,  # Allow residual if dims match
                        )
                    )
                    current_dim = next_dim
                
                # Final layer: last hidden dim -> output dim
                expansion_factor = getattr(self.config, 'expansion_factor', 2.0)
                hidden_dim = int(current_dim * expansion_factor)
                layers.append(
                    MLPBlock(
                        input_dim=current_dim,
                        hidden_dim=hidden_dim,
                        output_dim=self.output_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                        use_prenorm=use_prenorm,
                        norm_type=norm_type,
                        residual=False,  # Usually we don't add residual to final output
                    )
                )
        
        # Case 2: Use expansion_factor approach (backward compatible)
        else:
            if self.num_layers == 1:
                # Single block: input -> hidden -> output
                layers.append(
                    MLPBlock(
                        input_dim=self.input_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=self.output_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                        use_prenorm=use_prenorm,
                        norm_type=norm_type,
                        residual=False,
                    )
                )
            else:
                # First layer: input_dim -> hidden_dim
                layers.append(
                    MLPBlock(
                        input_dim=self.input_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                        use_prenorm=use_prenorm,
                        norm_type=norm_type,
                        residual=False,  # Can't have residual from different input dim
                    )
                )
                
                # Intermediate layers: hidden_dim -> hidden_dim (all same dimensions)
                # This allows residual connections
                for _ in range(self.num_layers - 2):
                    layers.append(
                        MLPBlock(
                            input_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            output_dim=self.hidden_dim,
                            use_swiglu=self.use_swiglu,
                            dropout_rate=dropout_rate,
                            use_prenorm=use_prenorm,
                            norm_type=norm_type,
                            residual=True,  # Add residual between identical dimensions
                        )
                    )
                
                # Last layer: hidden_dim -> output_dim
                layers.append(
                    MLPBlock(
                        input_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=self.output_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                        use_prenorm=use_prenorm,
                        norm_type=norm_type,
                        residual=False,  # Usually no residual to different output dim
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
        
        # Forward pass through all layers sequentially
        # Residual connections are now handled WITHIN each MLPBlock,
        # so we simply pass the output of one layer to the next
        out = x
        for layer in self.layers:
            out = layer(out)
        
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
        if self.hidden_layer_dims is not None:
            # Custom layer structure
            if len(self.hidden_layer_dims) == 0:
                layer_description = "Direct input -> output projection"
                total_layers = 1
            else:
                layer_dims = [self.input_dim] + self.hidden_layer_dims + [self.output_dim]
                layer_description = " -> ".join(str(d) for d in layer_dims)
                total_layers = len(self.hidden_layer_dims)
            
            info_lines = [
                "=" * 60,
                "Chemical Density Surrogate Model (Custom Layer Structure)",
                "=" * 60,
                f"Architecture: {layer_description}",
                f"Hidden layers: {total_layers}",
                f"Activation: {'SwiGLU' if self.use_swiglu else 'Swish'}",
                f"Total parameters: {self.get_num_parameters():,}",
                "=" * 60,
            ]
        else:
            # Expansion factor structure
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
