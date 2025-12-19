"""Optuna objective function for hyperparameter tuning.

This module provides an objective function for Optuna to optimize model
hyperparameters across all 3 architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import logging

from tune_config import OptunaSearchSpace
from config import ModelConfig, TrainingConfig
from model import ChemicalDensitySurrogate
from data_loader import ChemicalDensityDataLoader
from trainer import ModelTrainer

# Import optional architectures
try:
    from cnn_model import ConvolutionalSurrogate, MultiScaleConvolutionalSurrogate
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    from lightgbm_model import LightGBMSurrogate
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


def create_model_from_hyperparams(
    hyperparameters: Dict[str, Any],
    architecture: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """Create a model based on architecture and hyperparameters.
    
    Args:
        hyperparameters: Dictionary of hyperparameters
        architecture: Model architecture ("mlp", "cnn", "cnn_multiscale", "lightgbm")
        device: Device to use ("cuda" or "cpu")
        
    Returns:
        Instantiated model
    """
    dropout_rate = hyperparameters.get("dropout_rate", 0.2)
    
    if architecture == "mlp":
        model_config = ModelConfig(
            architecture="mlp",
            input_dim=4,
            output_dim=1,
            num_layers=int(hyperparameters.get("num_layers", 2)),
            expansion_factor=float(hyperparameters.get("expansion_factor", 4.0)),
            use_swiglu=bool(hyperparameters.get("use_swiglu", False)),
            dropout_rate=dropout_rate,
            device=device,
        )
        return ChemicalDensitySurrogate(model_config)
    
    elif architecture == "cnn":
        if not CNN_AVAILABLE:
            raise ImportError("CNN model not available. Check cnn_model.py")
        
        model_config = ModelConfig(
            architecture="cnn",
            input_dim=4,
            output_dim=1,
            cnn_expansion_size=int(hyperparameters.get("cnn_expansion_size", 8)),
            cnn_num_layers=int(hyperparameters.get("cnn_num_layers", 4)),
            cnn_kernel_size=int(hyperparameters.get("cnn_kernel_size", 3)),
            cnn_use_batch_norm=bool(hyperparameters.get("cnn_use_batch_norm", True)),
            cnn_use_residual=bool(hyperparameters.get("cnn_use_residual", False)),
            dropout_rate=dropout_rate,
            device=device,
        )
        return ConvolutionalSurrogate(
            num_input_features=model_config.input_dim,
            expansion_size=model_config.cnn_expansion_size,
            num_conv_layers=model_config.cnn_num_layers,
            kernel_size=model_config.cnn_kernel_size,
            use_batch_norm=model_config.cnn_use_batch_norm,
            use_residual=model_config.cnn_use_residual,
            dropout_rate=model_config.dropout_rate,
        )
    
    elif architecture == "cnn_multiscale":
        if not CNN_AVAILABLE:
            raise ImportError("CNN model not available. Check cnn_model.py")
        
        model_config = ModelConfig(
            architecture="cnn_multiscale",
            input_dim=4,
            output_dim=1,
            cnn_multiscale_expansion_size=int(hyperparameters.get("cnn_multiscale_expansion_size", 16)),
            cnn_multiscale_num_scales=int(hyperparameters.get("cnn_multiscale_num_scales", 3)),
            cnn_multiscale_base_channels=int(hyperparameters.get("cnn_multiscale_base_channels", 16)),
            dropout_rate=dropout_rate,
            device=device,
        )
        return MultiScaleConvolutionalSurrogate(
            num_input_features=model_config.input_dim,
            expansion_size=model_config.cnn_multiscale_expansion_size,
            num_scales=model_config.cnn_multiscale_num_scales,
            base_channels=model_config.cnn_multiscale_base_channels,
            dropout_rate=model_config.dropout_rate,
        )
    
    elif architecture == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        return LightGBMSurrogate(
            num_leaves=int(hyperparameters.get("lgb_num_leaves", 31)),
            learning_rate=float(hyperparameters.get("lgb_learning_rate", 0.05)),
            num_boost_round=int(hyperparameters.get("lgb_num_boost_round", 100)),
            max_depth=int(hyperparameters.get("lgb_max_depth", -1)),
            min_child_samples=int(hyperparameters.get("lgb_min_child_samples", 20)),
            subsample=float(hyperparameters.get("lgb_subsample", 0.8)),
            colsample_bytree=float(hyperparameters.get("lgb_colsample_bytree", 0.8)),
            reg_alpha=float(hyperparameters.get("lgb_reg_alpha", 0.0)),
            reg_lambda=float(hyperparameters.get("lgb_reg_lambda", 0.0)),
            boosting_type=hyperparameters.get("lgb_boosting_type", "gbdt"),
            metric="rmse",
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    hyperparameters: Dict[str, Any],
    num_epochs: int,
    device: str,
    target_std: float = None,
) -> Tuple[float, float, float, float]:
    """Train a model with given hyperparameters.
    
    Supports both PyTorch models (nn.Module) and scikit-learn compatible models (e.g., LightGBM).
    
    Args:
        model: PyTorch model (nn.Module) or scikit-learn compatible model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        hyperparameters: Dictionary of hyperparameters
        num_epochs: Number of epochs to train
        device: Device to use
        target_std: Target standard deviation for denormalization (if None, will try to get from dataset)
        
    Returns:
        Tuple of (best_val_rmse, test_rmse, test_rmse_denorm, test_mae)
    """
    # Create trainer
    trainer = ModelTrainer(model, device=device, checkpoint_dir="./optuna_checkpoints")
    
    # Detect if model is PyTorch or scikit-learn based
    is_pytorch = isinstance(model, nn.Module)
    
    if is_pytorch:
        # PyTorch training pipeline
        # Extract training hyperparameters
        learning_rate = float(hyperparameters.get("learning_rate", 0.01))
        weight_decay = float(hyperparameters.get("weight_decay", 1e-5))
        optimizer_name = hyperparameters.get("optimizer", "adam")
        lr_scheduler = hyperparameters.get("lr_scheduler", "cosine")
        loss_fn = hyperparameters.get("loss_fn", "mse")
        
        # Configure loss and optimizer
        criterion = trainer.configure_loss(loss_fn)
        optimizer = trainer.configure_optimizer(optimizer_name, learning_rate, weight_decay)
        scheduler = trainer.configure_scheduler(
            optimizer=optimizer,
            scheduler_name=lr_scheduler,
            num_epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            cosine_t_max=num_epochs,
            cosine_eta_min=1e-3,
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_rmse = trainer.train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler=scheduler,
                show_progress=False,
            )
            
            # Validate
            val_rmse = trainer.validate(val_loader, criterion)
            
            # Step scheduler
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            # Track best validation loss
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
        
        # Test on final model
        test_rmse, predictions, targets = trainer.test(test_loader, criterion)
        
        # Calculate MAE (on normalized scale)
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # Denormalize RMSE and MAE by multiplying by target_std
        # Formula: error_denorm = error_norm * target_std
        # (We multiply by std but don't add mean because error metrics don't have an offset)
        if target_std is None:
            target_std = test_loader.dataset.target_std if hasattr(test_loader.dataset, 'target_std') else 1.0
        
        test_rmse_denorm = test_rmse * target_std
        mae_denorm = mae * target_std
        
        return best_val_loss, test_rmse, test_rmse_denorm, mae_denorm
    
    else:
        # Scikit-learn compatible model training (e.g., LightGBM)
        # Train using trainer's sklearn method
        history = trainer._train_sklearn_model(
            train_loader, 
            val_loader, 
            num_epochs=num_epochs,
            show_progress_bar=False,
            save_best_model=True
        )
        
        # Get best validation RMSE from history
        val_rmse_list = history.get("val_loss", [])
        if val_rmse_list:
            best_val_rmse = float(min(val_rmse_list))
        else:
            best_val_rmse = float('inf')
        
        # Test the model - returns (rmse, predictions_tensor, targets_tensor)
        test_rmse, predictions, targets = trainer._test_sklearn_model(test_loader)
        
        # Convert to float for consistency
        test_rmse = float(test_rmse)
        
        # Calculate MAE (on normalized scale)
        test_mae = float(torch.mean(torch.abs(predictions - targets)).item())
        
        # Denormalize RMSE and MAE
        # Get target_std from passed parameter or test_loader dataset
        if target_std is None:
            target_std = test_loader.dataset.target_std if hasattr(test_loader.dataset, 'target_std') else 1.0
        
        test_rmse_denorm = test_rmse * target_std
        test_mae_denorm = test_mae * target_std
        
        return best_val_rmse, test_rmse, test_rmse_denorm, test_mae_denorm


def create_objective(
    architecture: str,
    num_epochs: int,
    data_dir: str,
    device: str,
    suggest_fn,
    seed: int = 46,
) -> callable:
    """Create an objective function for Optuna.
    
    Args:
        architecture: Model architecture
        num_epochs: Number of epochs to train
        data_dir: Directory with data
        device: Device to use
        suggest_fn: Function to suggest hyperparameters
        seed: Random seed for reproducibility (default: 46)
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial) -> float:
        """Objective function to minimize (validation RMSE)."""
        try:
            # Generate a unique seed for this trial (not fixed for grid search)
            trial_seed = OptunaSearchSpace.generate_trial_seed(seed, trial.number)
            
            # Set random seed for reproducibility within each trial
            import numpy as np
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(trial_seed)
                torch.cuda.manual_seed_all(trial_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Suggest hyperparameters
            hyperparameters = suggest_fn(trial)
            
            # Create model
            model = create_model_from_hyperparams(hyperparameters, architecture, device)
            
            # Load data
            data_loader = ChemicalDensityDataLoader(data_dir)
            batch_size = int(hyperparameters.get("batch_size", 64))
            train_loader, val_loader, test_loader, dataset = data_loader.load_dataset(
                normalize_features=True,
                normalize_targets=True,
                validation_split=0.2,
                test_split=0.05,
                batch_size=batch_size,
            )
            
            # Train model
            best_val_rmse, test_rmse, test_rmse_denorm, test_mae = train_model(
                model,
                train_loader,
                val_loader,
                test_loader,
                hyperparameters,
                num_epochs,
                device,
                target_std=dataset.target_std,  # Pass the correct target_std
            )
            
            # Check for invalid results
            if best_val_rmse is None or best_val_rmse == float('inf') or best_val_rmse != best_val_rmse:  # NaN check
                logger.warning(f"Trial returned invalid RMSE: {best_val_rmse}")
                return float('inf')
            
            # Log trial info
            trial.set_user_attr("test_rmse", float(test_rmse) if test_rmse != float('inf') else None)
            trial.set_user_attr("test_rmse_denorm", float(test_rmse_denorm) if test_rmse_denorm != float('inf') else None)
            trial.set_user_attr("test_mae", float(test_mae) if test_mae != float('inf') else None)
            trial.set_user_attr("target_std", float(dataset.target_std) if hasattr(dataset, 'target_std') else None)
            
            # Debug: Log validation RMSE denormalization
            val_rmse_denorm = best_val_rmse * (float(dataset.target_std) if hasattr(dataset, 'target_std') else 1.0)
            trial.set_user_attr("val_rmse_denorm", float(val_rmse_denorm))
            
            # CRITICAL: Return validation RMSE for optimization (not test RMSE)
            # Test set should ONLY be used for final evaluation, never for tuning
            # Using test RMSE for optimization would overfit hyperparameters to test set
            # Validation set is the appropriate metric for hyperparameter selection
            logger.info(f"Trial {trial.number}: Val RMSE={best_val_rmse:.6f}, Test RMSE={test_rmse:.6f}")
            return float(best_val_rmse)
        
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            # Return a large value to indicate failure
            return float('inf')
    
    return objective


if __name__ == "__main__":
    print("Optuna objective function loaded. Use tune.py to run hyperparameter tuning.")
