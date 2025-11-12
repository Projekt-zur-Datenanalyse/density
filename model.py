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
        """Build the network layers.
        
        Returns:
            List of layers in the network
        """
        layers = nn.ModuleList()
        
        # Get dropout rate from config if available
        dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
        
        # Case 1: Custom hidden layer dimensions provided
        if self.hidden_layer_dims is not None:
            if len(self.hidden_layer_dims) == 0:
                # Empty array: direct projection from input to output
                layers.append(
                    MLPBlock(
                        input_dim=self.input_dim,
                        hidden_dim=self.input_dim,  # Not used in this case but required
                        output_dim=self.output_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                    )
                )
            else:
                # One or more hidden layers with specified dimensions
                # First layer: input -> first hidden
                layers.append(
                    MLPBlock(
                        input_dim=self.input_dim,
                        hidden_dim=self.hidden_layer_dims[0],
                        output_dim=self.hidden_layer_dims[0],
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                    )
                )
                
                # Intermediate layers: hidden -> hidden
                for i in range(len(self.hidden_layer_dims) - 1):
                    current_dim = self.hidden_layer_dims[i]
                    next_dim = self.hidden_layer_dims[i + 1]
                    layers.append(
                        MLPBlock(
                            input_dim=current_dim,
                            hidden_dim=next_dim,
                            output_dim=next_dim,
                            use_swiglu=self.use_swiglu,
                            dropout_rate=dropout_rate,
                        )
                    )
                
                # Last layer: last hidden -> output
                layers.append(
                    MLPBlock(
                        input_dim=self.hidden_layer_dims[-1],
                        hidden_dim=self.hidden_layer_dims[-1],
                        output_dim=self.output_dim,
                        use_swiglu=self.use_swiglu,
                        dropout_rate=dropout_rate,
                    )
                )
        
        # Case 2: Use expansion_factor approach (backward compatible)
        else:
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
        
        # Case 1: Custom hidden layer dimensions
        if self.hidden_layer_dims is not None:
            if len(self.hidden_layer_dims) == 0:
                # Direct projection: input -> output
                return self.layers[0](x)
            else:
                # One or more hidden layers
                # Note: Residual connections only work when dimensions match
                # For custom dims, only apply residuals where consecutive dims are identical
                
                out = self.layers[0](x)  # First layer: input -> first hidden
                
                # Apply intermediate layers with residual connections where possible
                for i, layer in enumerate(self.layers[1:-1], 1):
                    prev_out_dim = self.hidden_layer_dims[i - 1]
                    curr_out_dim = self.hidden_layer_dims[i]
                    
                    # Only add residual connection if dimensions match
                    if prev_out_dim == curr_out_dim:
                        out = out + layer(out)
                    else:
                        out = layer(out)
                
                # Last layer: last hidden -> output
                out = self.layers[-1](out)
                
                return out
        
        # Case 2: Expansion factor approach (backward compatible)
        else:
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
