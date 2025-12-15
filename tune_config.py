"""Hyperparameter search space configurations for Optuna.

This module defines search spaces for all 3 architectures:
- MLP: Multi-Layer Perceptron
- CNN: Convolutional Neural Network
- CNN_MULTISCALE: Multi-Scale CNN (Inception-style)

Uses Optuna's trial suggestion API for hyperparameter optimization.
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable
import optuna
from optuna.trial import Trial


@dataclass
class OptunaSearchSpace:
    """Base class for defining hyperparameter search spaces for Optuna."""
    
    @staticmethod
    def suggest_mlp_hyperparameters(trial: Trial) -> Dict[str, Any]:
        """Suggest MLP hyperparameters for an Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters with fixed 5-layer architecture
        """
        base = trial.suggest_categorical("base_dim", [16, 32, 64, 128, 256])
        expand_factor = trial.suggest_float("expand_factor", 2, 6)
        compress_factor = trial.suggest_categorical("compress_factor", [2, 4, 8])

        h1 = base
        h2 = int(base * expand_factor)
        h3 = base
        h4 = max(base // compress_factor, 8)

        return {
            "hidden_layer_dims": [h1, h2, h3, h4],
            "use_swiglu": trial.suggest_categorical("use_swiglu", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.2),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "optimizer": "adam",
            "lr_scheduler": "cosine",
            "loss_fn": trial.suggest_categorical("loss_fn", ["mse", "mae"]),
        }

    
    @staticmethod
    def suggest_cnn_hyperparameters(trial: Trial) -> Dict[str, Any]:
        """Suggest CNN hyperparameters for an Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        return {
            # CNN architecture
            "cnn_expansion_size": trial.suggest_categorical("cnn_expansion_size", [4, 8, 12, 16]),
            "cnn_num_layers": trial.suggest_int("cnn_num_layers", 2, 6),
            "cnn_kernel_size": trial.suggest_categorical("cnn_kernel_size", [3, 5, 7]),
            "cnn_use_batch_norm": trial.suggest_categorical("cnn_use_batch_norm", [False, True]),
            "cnn_use_residual": trial.suggest_categorical("cnn_use_residual", [False, True]),
            
            # Training hyperparameters (common)
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            
            # Optimizer and scheduler
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "onecycle", "cosine"]),
            
            # Loss function
            "loss_fn": trial.suggest_categorical("loss_fn", ["mse", "mae"]),
        }
    
    @staticmethod
    def suggest_cnn_multiscale_hyperparameters(trial: Trial) -> Dict[str, Any]:
        """Suggest Multi-Scale CNN hyperparameters for an Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        return {
            # Multi-Scale CNN architecture
            "cnn_multiscale_expansion_size": trial.suggest_categorical("cnn_multiscale_expansion_size", [8, 12, 16, 20]),
            "cnn_multiscale_num_scales": trial.suggest_int("cnn_multiscale_num_scales", 2, 6),
            "cnn_multiscale_base_channels": trial.suggest_categorical("cnn_multiscale_base_channels", [8, 16, 32, 64]),
            
            # Training hyperparameters (common)
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            
            # Optimizer and scheduler
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "onecycle", "cosine"]),
            
            # Loss function
            "loss_fn": trial.suggest_categorical("loss_fn", ["mse"]),
        }
    
    @staticmethod
    def suggest_lightgbm_hyperparameters(trial: Trial) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters for an Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        return {
            # LightGBM tree architecture
            "lgb_num_leaves": trial.suggest_categorical("lgb_num_leaves", [7, 15, 31, 63, 127, 255]),
            "lgb_max_depth": trial.suggest_categorical("lgb_max_depth", [3, 5, 7, 10, 15, -1]),  # -1 for unlimited
            "lgb_min_child_samples": trial.suggest_categorical("lgb_min_child_samples", [5, 10, 20, 30, 50]),
            
            # LightGBM boosting
            "lgb_learning_rate": trial.suggest_float("lgb_learning_rate", 0.001, 0.3, log=True),
            "lgb_num_boost_round": trial.suggest_categorical("lgb_num_boost_round", [50, 100, 150, 200, 300]),
            
            # LightGBM regularization
            # Note: Using linear scale for reg_alpha/lambda since they can be 0
            # For log scale, low must be > 0, so we use 1e-6 as practical lower bound with log scale
            "lgb_reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-6, 10.0, log=True),
            "lgb_reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-6, 10.0, log=True),
            "lgb_subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "lgb_colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.5, 1.0),
            
            # LightGBM boosting type
            "lgb_boosting_type": trial.suggest_categorical("lgb_boosting_type", ["gbdt", "dart", "goss"]),
            
            # Batch processing (for data loaders compatibility)
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }
    
    @staticmethod
    def get_suggest_function(architecture: str) -> Callable[[Trial], Dict[str, Any]]:
        """Get suggestion function for a specific architecture.
        
        Args:
            architecture: Architecture name ("mlp", "cnn", "cnn_multiscale", or "lightgbm")
            
        Returns:
            Callable that suggests hyperparameters for the architecture
        """
        functions = {
            "mlp": OptunaSearchSpace.suggest_mlp_hyperparameters,
            "cnn": OptunaSearchSpace.suggest_cnn_hyperparameters,
            "cnn_multiscale": OptunaSearchSpace.suggest_cnn_multiscale_hyperparameters,
            "lightgbm": OptunaSearchSpace.suggest_lightgbm_hyperparameters,
        }
        
        if architecture not in functions:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Must be one of {list(functions.keys())}"
            )
        
        return functions[architecture]

    @staticmethod
    def get_default_config(architecture: str) -> Dict[str, Any]:
        """Get default configuration for a specific architecture.
        
        These defaults are based on config.py and are used as baselines.
        
        Args:
            architecture: Architecture name
            
        Returns:
            Dictionary with default hyperparameter values
        """
        defaults = {
            "mlp": {
                "hidden_layer_dims": [32, 16, 8, 4],  # 4 hidden layers with decreasing dimensions
                "use_swiglu": False,
                "learning_rate": 0.01,
                "batch_size": 64,
                "dropout_rate": 0.2,
                "weight_decay": 1e-5,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "loss_fn": "mse",
            },
            "cnn": {
                "cnn_expansion_size": 8,
                "cnn_num_layers": 4,
                "cnn_kernel_size": 3,
                "cnn_use_batch_norm": True,
                "cnn_use_residual": False,
                "learning_rate": 0.01,
                "batch_size": 64,
                "dropout_rate": 0.2,
                "weight_decay": 1e-5,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "loss_fn": "mse",
            },
            "cnn_multiscale": {
                "cnn_multiscale_expansion_size": 16,
                "cnn_multiscale_num_scales": 3,
                "cnn_multiscale_base_channels": 16,
                "learning_rate": 0.01,
                "batch_size": 64,
                "dropout_rate": 0.2,
                "weight_decay": 1e-5,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "loss_fn": "mse",
            },
            "lightgbm": {
                "lgb_num_leaves": 31,
                "lgb_max_depth": -1,
                "lgb_min_child_samples": 20,
                "lgb_learning_rate": 0.05,
                "lgb_num_boost_round": 100,
                "lgb_reg_alpha": 0.0,
                "lgb_reg_lambda": 0.0,
                "lgb_subsample": 0.8,
                "lgb_colsample_bytree": 0.8,
                "lgb_boosting_type": "gbdt",
                "batch_size": 64,
            },
        }
        
        if architecture not in defaults:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Must be one of {list(defaults.keys())}"
            )
        
        return defaults[architecture]


# Predefined search configurations for Optuna studies
# These can be used for quick experimentation without modifying the code

SEARCH_SPACE_CONFIGS = {
    "minimal": {
        "description": "Minimal search space - quick testing",
        "n_trials": 10,
        "max_epochs": 20,
    },
    "balanced": {
        "description": "Balanced search space - good exploration/exploitation tradeoff",
        "n_trials": 50,
        "max_epochs": 50,
    },
    "extensive": {
        "description": "Extensive search space - thorough hyperparameter exploration",
        "n_trials": 100,
        "max_epochs": 100,
    },
    "production": {
        "description": "Production search space - comprehensive search with full training",
        "n_trials": 2500,
        "max_epochs": 150,
    },
}


def get_search_config(config_type: str = "balanced") -> Dict[str, Any]:
    """Get predefined search configuration.
    
    Args:
        config_type: Type of configuration ("minimal", "balanced", "extensive", "production")
        
    Returns:
        Dictionary with search configuration
    """
    if config_type not in SEARCH_SPACE_CONFIGS:
        raise ValueError(
            f"Unknown config type '{config_type}'. "
            f"Must be one of {list(SEARCH_SPACE_CONFIGS.keys())}"
        )
    
    return SEARCH_SPACE_CONFIGS[config_type]


if __name__ == "__main__":
    # Example usage
    print("Available architectures:", ["mlp", "cnn", "cnn_multiscale", "gnn"])
    print("\nMLP Default Config:")
    print(OptunaSearchSpace.get_default_config("mlp"))
    print("\nAvailable search configs:", list(SEARCH_SPACE_CONFIGS.keys()))
