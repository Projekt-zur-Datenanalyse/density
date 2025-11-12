"""Optuna objective function for hyperparameter tuning.

This module provides an objective function for Optuna to optimize model
hyperparameters across all 4 architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import logging

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
    from gnn_model import GraphNeuralSurrogate
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

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
        architecture: Model architecture ("mlp", "cnn", "cnn_multiscale", "gnn")
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
    
    elif architecture == "gnn":
        if not GNN_AVAILABLE:
            raise ImportError("GNN model not available. Check gnn_model.py")
        
        model_config = ModelConfig(
            architecture="gnn",
            input_dim=4,
            output_dim=1,
            gnn_hidden_dim=int(hyperparameters.get("gnn_hidden_dim", 16)),
            gnn_num_layers=int(hyperparameters.get("gnn_num_layers", 6)),
            gnn_type=hyperparameters.get("gnn_type", "gat"),
            dropout_rate=dropout_rate,
            device=device,
        )
        return GraphNeuralSurrogate(
            num_node_features=model_config.input_dim,
            hidden_dim=model_config.gnn_hidden_dim,
            num_layers=model_config.gnn_num_layers,
            gnn_type=model_config.gnn_type,
            dropout_rate=model_config.dropout_rate,
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    hyperparameters: Dict[str, Any],
    num_epochs: int,
    device: str,
) -> Tuple[float, float, float, float]:
    """Train a model with given hyperparameters.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        hyperparameters: Dictionary of hyperparameters
        num_epochs: Number of epochs to train
        device: Device to use
        
    Returns:
        Tuple of (best_val_rmse, test_rmse, test_rmse_denorm, test_mae)
    """
    # Create trainer
    trainer = ModelTrainer(model, device=device, checkpoint_dir="./optuna_checkpoints")
    
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
    test_rmse_denorm = test_rmse * test_loader.dataset.target_std if hasattr(test_loader.dataset, 'target_std') else test_rmse
    mae_denorm = mae * test_loader.dataset.target_std if hasattr(test_loader.dataset, 'target_std') else mae
    
    return best_val_loss, test_rmse, test_rmse_denorm, mae_denorm


def create_objective(
    architecture: str,
    num_epochs: int,
    data_dir: str,
    device: str,
    suggest_fn,
) -> callable:
    """Create an objective function for Optuna.
    
    Args:
        architecture: Model architecture
        num_epochs: Number of epochs to train
        data_dir: Directory with data
        device: Device to use
        suggest_fn: Function to suggest hyperparameters
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial) -> float:
        """Objective function to minimize (validation RMSE)."""
        try:
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
                test_split=0.1,
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
            )
            
            # Log trial info
            trial.set_user_attr("test_rmse", test_rmse)
            trial.set_user_attr("test_rmse_denorm", test_rmse_denorm)
            trial.set_user_attr("test_mae", test_mae)
            trial.set_user_attr("target_std", float(dataset.target_std) if hasattr(dataset, 'target_std') else None)
            
            return best_val_rmse
        
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a large value to indicate failure
            return float('inf')
    
    return objective


if __name__ == "__main__":
    print("Optuna objective function loaded. Use tune.py to run hyperparameter tuning.")
