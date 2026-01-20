"""
Optuna objective function for hyperparameter tuning.

This module provides objective functions that Optuna uses to
evaluate different hyperparameter configurations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Callable
import logging

from .search_spaces import SearchSpace
from core import DataLoader, Trainer, set_seed, get_device
from models import create_model

logger = logging.getLogger(__name__)


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    hyperparams: Dict[str, Any],
    num_epochs: int,
    device: str,
    target_std: float,
) -> Tuple[float, float, float, float]:
    """Train a model and evaluate it.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        hyperparams: Hyperparameters for training
        num_epochs: Number of epochs
        device: Device to use
        target_std: Target standard deviation for denormalization
        
    Returns:
        Tuple of (best_val_rmse, test_rmse, test_rmse_denorm, test_mae_denorm)
    """
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir="./tuning_checkpoints",
    )
    
    # Get training hyperparameters
    learning_rate = float(hyperparams.get("learning_rate", 0.01))
    weight_decay = float(hyperparams.get("weight_decay", 1e-5))
    optimizer_name = hyperparams.get("optimizer", "adam")
    scheduler_name = hyperparams.get("lr_scheduler", "cosine")
    loss_fn = hyperparams.get("loss_fn", "mse")
    
    # Train (silently)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        loss_fn=loss_fn,
        show_progress=False,
        save_best=True,
        verbose=0,
    )
    
    # Test
    test_results = trainer.test(test_loader, target_std=target_std)
    
    return (
        trainer.best_val_loss,
        test_results['rmse_normalized'],
        test_results.get('rmse_denormalized', test_results['rmse_normalized'] * target_std),
        test_results.get('mae_denormalized', test_results['mae_normalized'] * target_std),
    )


def create_objective(
    architecture: str,
    num_epochs: int,
    data_path: str,
    device: str,
    suggest_fn: Callable,
    seed: int = 46,
) -> Callable:
    """Create an objective function for Optuna.
    
    Args:
        architecture: Model architecture
        num_epochs: Number of epochs to train
        data_path: Path to dataset
        device: Device to use
        suggest_fn: Function to suggest hyperparameters
        seed: Master seed for reproducibility
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial) -> float:
        """Objective function to minimize (validation RMSE)."""
        try:
            # Generate unique seed for this trial
            trial_seed = SearchSpace.generate_trial_seed(seed, trial.number)
            set_seed(trial_seed)
            
            # Suggest hyperparameters
            hyperparams = suggest_fn(trial)
            
            # Load data
            data_loader = DataLoader(data_path)
            batch_size = int(hyperparams.get("batch_size", 64))
            
            train_loader, val_loader, test_loader, stats = data_loader.load(
                validation_split=0.15,
                test_split=0.10,
                batch_size=batch_size,
                seed=trial_seed,
            )
            
            target_std = stats['normalization']['target_std']
            
            # Create model
            model = create_model(architecture, hyperparams)
            
            # Train and evaluate
            best_val_rmse, test_rmse, test_rmse_denorm, test_mae_denorm = train_and_evaluate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                hyperparams=hyperparams,
                num_epochs=num_epochs,
                device=device,
                target_std=target_std,
            )
            
            # Store additional metrics as user attributes
            trial.set_user_attr("test_rmse", test_rmse)
            trial.set_user_attr("test_rmse_denorm", test_rmse_denorm)
            trial.set_user_attr("test_mae_denorm", test_mae_denorm)
            trial.set_user_attr("trial_seed", trial_seed)
            
            logger.info(
                f"Trial {trial.number}: val_rmse={best_val_rmse:.6f}, "
                f"test_rmse={test_rmse_denorm:.2f} kg/m³"
            )
            
            return best_val_rmse
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise
    
    return objective


__all__ = ["create_objective", "train_and_evaluate"]
