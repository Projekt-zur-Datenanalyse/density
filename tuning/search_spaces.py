"""
Hyperparameter search space definitions for Optuna.

This module defines search spaces for all architectures:
- MLP: Multi-Layer Perceptron
- CNN: Convolutional Neural Network  
- CNN_MULTISCALE: Multi-Scale CNN (Inception-style)
- LightGBM: Gradient Boosting Machine

Uses Optuna's trial suggestion API for hyperparameter sampling.
"""

from typing import Dict, Any, Callable
import numpy as np
from optuna.trial import Trial


class SearchSpace:
    """Hyperparameter search space definitions for Optuna."""
    
    @staticmethod
    def generate_trial_seed(master_seed: int, trial_number: int) -> int:
        """Generate a unique seed for a specific trial.
        
        This ensures different trials get different random initializations,
        especially important for grid search where sampler seed is not used.
        
        Args:
            master_seed: Base seed for reproducibility
            trial_number: Trial number from Optuna
            
        Returns:
            Unique seed for this trial
        """
        rng = np.random.RandomState(master_seed + trial_number * 1000)
        return int(rng.randint(0, 2**31 - 1))
    
    @staticmethod
    def suggest_mlp(trial: Trial) -> Dict[str, Any]:
        """Suggest MLP hyperparameters.
        
        Uses flexible layer architecture:
        - 1 layer: [expand]
        - 2 layers: [expand, compress]
        - 3 layers: [expand, base, compress]
        - 4 layers: [base, expand, base, compress]
        - 5 layers: [base, expand, base, base, compress]
        """
        base = trial.suggest_categorical("base_dim", [16, 32, 64, 128, 256])
        expand_factor = trial.suggest_float("expand_factor", 2, 6)
        compress_factor = trial.suggest_categorical("compress_factor", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 5)

        expand_dim = int(base * expand_factor)
        compress_dim = max(base // compress_factor, 8)
        
        # Construct hidden layer dims based on num_layers
        if num_layers == 1:
            hidden_dims = [expand_dim]
        elif num_layers == 2:
            hidden_dims = [expand_dim, compress_dim]
        elif num_layers == 3:
            hidden_dims = [expand_dim, base, compress_dim]
        elif num_layers == 4:
            hidden_dims = [base, expand_dim, base, compress_dim]
        else:  # 5 layers
            hidden_dims = [base, expand_dim, base, base, compress_dim]

        return {
            "hidden_dims": hidden_dims,
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
    def suggest_cnn(trial: Trial) -> Dict[str, Any]:
        """Suggest CNN hyperparameters."""
        return {
            # CNN architecture
            "expansion_size": trial.suggest_categorical("expansion_size", [4, 8, 12, 16]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [False, True]),
            "use_residual": trial.suggest_categorical("use_residual", [False, True]),
            
            # Training hyperparameters
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
    def suggest_cnn_multiscale(trial: Trial) -> Dict[str, Any]:
        """Suggest Multi-Scale CNN hyperparameters."""
        return {
            # Multi-Scale CNN architecture
            "expansion_size": trial.suggest_categorical("expansion_size", [8, 12, 16, 20]),
            "num_scales": trial.suggest_int("num_scales", 2, 6),
            "base_channels": trial.suggest_categorical("base_channels", [8, 16, 32, 64]),
            
            # Training hyperparameters
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
    def suggest_lightgbm(trial: Trial) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters."""
        return {
            # Tree architecture
            "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127, 255]),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, 15, -1]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 10, 20, 30, 50]),
            
            # Boosting
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "num_boost_round": trial.suggest_categorical("num_boost_round", [50, 100, 150, 200, 300]),
            
            # Regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            
            # Boosting type
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            
            # Batch processing (for data loader compatibility)
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }
    
    @staticmethod
    def get_suggest_function(architecture: str) -> Callable[[Trial], Dict[str, Any]]:
        """Get suggestion function for a specific architecture.
        
        Args:
            architecture: Architecture name
            
        Returns:
            Callable that suggests hyperparameters
        """
        functions = {
            "mlp": SearchSpace.suggest_mlp,
            "cnn": SearchSpace.suggest_cnn,
            "cnn_multiscale": SearchSpace.suggest_cnn_multiscale,
            "lightgbm": SearchSpace.suggest_lightgbm,
        }
        
        if architecture not in functions:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Must be one of {list(functions.keys())}"
            )
        
        return functions[architecture]
    
    @staticmethod
    def get_default_config(architecture: str) -> Dict[str, Any]:
        """Get default configuration for a specific architecture."""
        defaults = {
            "mlp": {
                "hidden_dims": [256, 64, 32],
                "use_swiglu": False,
                "dropout_rate": 0.1,
                "learning_rate": 0.001,
                "batch_size": 64,
                "weight_decay": 1e-5,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "loss_fn": "mse",
            },
            "cnn": {
                "expansion_size": 8,
                "num_layers": 4,
                "kernel_size": 3,
                "use_batch_norm": False,
                "use_residual": True,
                "dropout_rate": 0.2,
                "learning_rate": 0.01,
                "batch_size": 64,
                "weight_decay": 1e-5,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "loss_fn": "mse",
            },
            "cnn_multiscale": {
                "expansion_size": 16,
                "num_scales": 3,
                "base_channels": 16,
                "dropout_rate": 0.2,
                "learning_rate": 0.01,
                "batch_size": 64,
                "weight_decay": 1e-5,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "loss_fn": "mse",
            },
            "lightgbm": {
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "learning_rate": 0.05,
                "num_boost_round": 100,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "boosting_type": "gbdt",
                "batch_size": 64,
            },
        }
        
        if architecture not in defaults:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return defaults[architecture]


# Predefined search configurations
SEARCH_CONFIGS = {
    "minimal": {
        "description": "Quick testing (10 trials, 20 epochs)",
        "n_trials": 10,
        "max_epochs": 20,
    },
    "balanced": {
        "description": "Standard tuning (50 trials, 50 epochs)",
        "n_trials": 50,
        "max_epochs": 50,
    },
    "extensive": {
        "description": "Comprehensive search (100 trials, 100 epochs)",
        "n_trials": 100,
        "max_epochs": 100,
    },
    "production": {
        "description": "Production search (200 trials, 150 epochs)",
        "n_trials": 200,
        "max_epochs": 150,
    },
}


def get_search_config(config_type: str) -> Dict[str, Any]:
    """Get predefined search configuration.
    
    Args:
        config_type: Type of configuration
        
    Returns:
        Dictionary with search configuration
    """
    if config_type not in SEARCH_CONFIGS:
        raise ValueError(
            f"Unknown config type '{config_type}'. "
            f"Must be one of {list(SEARCH_CONFIGS.keys())}"
        )
    
    return SEARCH_CONFIGS[config_type]


__all__ = ["SearchSpace", "SEARCH_CONFIGS", "get_search_config"]
