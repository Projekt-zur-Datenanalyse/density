"""
Models package for Chemical Density Surrogate.

This module provides a unified interface for all model architectures:
- MLP: Multi-Layer Perceptron
- CNN: Convolutional Neural Network  
- MultiScaleCNN: Multi-Scale CNN (Inception-style)
- LightGBM: Gradient Boosting (requires lightgbm package)

Usage:
    from models import create_model, AVAILABLE_ARCHITECTURES
    
    model = create_model("mlp", config)
"""

from typing import Dict, Any, List, Optional
import torch.nn as nn

# Import architectures
from .mlp import MLP, ACTIVATION_TYPES
from .cnn import CNN, MultiScaleCNN
from .activations import SwiGLU
from .kan import KANRegressor
from .siren import SIREN

# LightGBM is optional
try:
    from .lightgbm_model import LightGBMModel
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LightGBMModel = None

# Available architectures
AVAILABLE_ARCHITECTURES = ["mlp", "cnn", "cnn_multiscale"]
if LIGHTGBM_AVAILABLE:
    AVAILABLE_ARCHITECTURES.append("lightgbm")

AVAILABLE_ARCHITECTURES.extend(["kan", "siren"])


def create_model(
    architecture: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> nn.Module:
    """Create a model instance based on architecture type.
    
    Args:
        architecture: One of 'mlp', 'cnn', 'cnn_multiscale', 'lightgbm'
        config: Optional configuration dictionary. If None, uses default config.
        **kwargs: Additional keyword arguments passed to the model constructor.
                  These override config values if both are provided.
    
    Returns:
        Model instance
        
    Raises:
        ValueError: If architecture is not supported
        ImportError: If lightgbm is requested but not installed
        
    Example:
        # Using kwargs directly
        model = create_model("mlp", hidden_dims=[16, 32, 8])
        
        # Using config dict
        model = create_model("cnn", config={"expansion_size": 8, "num_layers": 4})
    """
    config = config or {}
    merged_config = {**config, **kwargs}
    
    if architecture == "mlp":
        return MLP(
            input_dim=merged_config.get("input_dim", 4),
            output_dim=merged_config.get("output_dim", 1),
            hidden_dims=merged_config.get("hidden_dims", [16, 32, 8]),
            activation=merged_config.get("activation", "relu"),
            dropout_rate=merged_config.get("dropout_rate", 0.0),
        )
    
    elif architecture == "cnn":
        return CNN(
            input_dim=merged_config.get("input_dim", 4),
            output_dim=merged_config.get("output_dim", 1),
            expansion_size=merged_config.get("expansion_size", 8),
            num_layers=merged_config.get("num_layers", 4),
            kernel_size=merged_config.get("kernel_size", 3),
            use_batch_norm=merged_config.get("use_batch_norm", False),
            use_residual=merged_config.get("use_residual", True),
            dropout_rate=merged_config.get("dropout_rate", 0.2),
        )
    
    elif architecture == "cnn_multiscale":
        return MultiScaleCNN(
            input_dim=merged_config.get("input_dim", 4),
            output_dim=merged_config.get("output_dim", 1),
            expansion_size=merged_config.get("expansion_size", 16),
            num_scales=merged_config.get("num_scales", 3),
            base_channels=merged_config.get("base_channels", 16),
            dropout_rate=merged_config.get("dropout_rate", 0.2),
        )
    
    elif architecture == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        return LightGBMModel(
            num_leaves=merged_config.get("num_leaves", 31),
            learning_rate=merged_config.get("learning_rate", 0.05),
            num_boost_round=merged_config.get("num_boost_round", 100),
            max_depth=merged_config.get("max_depth", -1),
            min_child_samples=merged_config.get("min_child_samples", 20),
            subsample=merged_config.get("subsample", 0.8),
            colsample_bytree=merged_config.get("colsample_bytree", 0.8),
            reg_alpha=merged_config.get("reg_alpha", 0.0),
            reg_lambda=merged_config.get("reg_lambda", 0.0),
            boosting_type=merged_config.get("boosting_type", "gbdt"),
        )
    
    elif architecture == "kan":
        return KANRegressor(
            input_dim=merged_config.get("input_dim", 4),
            output_dim=merged_config.get("output_dim", 1),
            hidden_dims=merged_config.get("hidden_dims", [16, 16]),
            grid_min=merged_config.get("grid_min", -2.0),
            grid_max=merged_config.get("grid_max", 2.0),
            num_grids=merged_config.get("num_grids", 5),
            use_base_update=merged_config.get("use_base_update", True),
            base_activation=merged_config.get("base_activation", "silu"),
            spline_weight_init_scale=merged_config.get("spline_weight_init_scale", 0.1),
        )

    elif architecture == "siren":
        return SIREN(
            input_dim=merged_config.get("input_dim", 4),
            output_dim=merged_config.get("output_dim", 1),
            hidden_dims=merged_config.get("hidden_dims", [64, 64, 64]),
            first_omega_0=merged_config.get("first_omega_0", 3.0),
            hidden_omega_0=merged_config.get("hidden_omega_0", 3.0),
            use_bias=merged_config.get("use_bias", True),
            outermost_linear=merged_config.get("outermost_linear", True),
        )
    
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {AVAILABLE_ARCHITECTURES}"
        )


def get_model_info(model) -> str:
    """Get model information string.
    
    Args:
        model: Model instance
        
    Returns:
        Formatted string with model details
    """
    if hasattr(model, 'get_model_info'):
        return model.get_model_info()
    elif hasattr(model, 'parameters'):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"Model with {num_params:,} trainable parameters"
    else:
        return str(model)


__all__ = [
    # Factory function
    "create_model",
    "get_model_info",
    # Constants
    "AVAILABLE_ARCHITECTURES",
    "ACTIVATION_TYPES",
    "LIGHTGBM_AVAILABLE",
    # Model classes
    "MLP",
    "CNN",
    "MultiScaleCNN",
    "LightGBMModel",
    # Building blocks
    "SwiGLU",
    # Experimental models
    "KANRegressor",
    "SIREN",
]
