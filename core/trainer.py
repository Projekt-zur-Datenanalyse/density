"""
Training utilities and trainer class.

This module provides the main Trainer class that handles:
- Training loop with progress tracking
- Validation
- Model checkpointing
- Learning rate scheduling
- Both PyTorch and LightGBM models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import numpy as np
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class Trainer:
    """Trainer for chemical density surrogate models.
    
    Handles both PyTorch models (nn.Module) and scikit-learn compatible
    models (e.g., LightGBM) transparently.
    """
    
    def __init__(
        self,
        model,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train (PyTorch nn.Module or sklearn-compatible)
            device: Device to use ("auto", "cuda", or "cpu")
            checkpoint_dir: Directory for saving checkpoints (None = no saving)
        """
        from .utils import get_device
        
        self.model = model
        self.device = get_device(device)
        
        # Set up checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Detect model type
        self.is_pytorch = isinstance(model, nn.Module)
        
        if self.is_pytorch:
            self.model.to(self.device)
        
        # Training state
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adam",
        scheduler_name: str = "cosine",
        loss_fn: str = "mse",
        show_progress: bool = True,
        save_best: bool = True,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            optimizer_name: Optimizer ("adam" or "sgd")
            scheduler_name: LR scheduler ("none", "cosine", "onecycle")
            loss_fn: Loss function ("mse" or "mae")
            show_progress: Show progress bars
            save_best: Save best model checkpoint
            verbose: Verbosity level (0=silent, 1=epochs, 2=detailed)
        
        Returns:
            Training history dictionary
        """
        # Handle non-PyTorch models
        if not self.is_pytorch:
            return self._train_sklearn_model(
                train_loader, val_loader, num_epochs, show_progress, save_best, verbose
            )
        
        # Configure training components
        optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)
        criterion = self._create_criterion(loss_fn)
        scheduler = self._create_scheduler(
            optimizer, scheduler_name, num_epochs, len(train_loader)
        )
        
        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Training on {self.device}")
            print(f"{'='*70}")
            print(f"Optimizer: {optimizer_name} (lr={learning_rate})")
            print(f"Scheduler: {scheduler_name}")
            print(f"Loss: {loss_fn}")
            print(f"Epochs: {num_epochs}")
            print(f"{'='*70}\n")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, show_progress
            )
            
            # Validate
            val_loss = self._validate(val_loader, criterion)
            
            # Step epoch-based schedulers
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Update best validation loss and save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_best and self.checkpoint_dir:
                    self._save_checkpoint(epoch + 1, is_best=True)
            
            # Print progress
            if verbose >= 1:
                msg = f"Epoch {epoch+1:3d}/{num_epochs} | "
                msg += f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                msg += f"LR: {current_lr:.2e}"
                
                if val_loss == self.best_val_loss:
                    msg += " *"
                
                print(msg)
        
        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Training complete! Best validation loss: {self.best_val_loss:.6f}")
            print(f"{'='*70}\n")
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        criterion,
        scheduler,
        show_progress: bool,
    ) -> float:
        """Train for one epoch.
        
        Returns:
            RMSE (root mean square error) computed on all training samples.
        """
        self.model.train()
        all_predictions = []
        all_targets = []
        
        iterator = train_loader
        if show_progress and HAS_TQDM:
            iterator = tqdm(train_loader, leave=False, desc="Training")
        
        for features, targets in iterator:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(features)
            loss = criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Store predictions for RMSE calculation
            all_predictions.append(predictions.detach())
            all_targets.append(targets)
            
            # Step OneCycleLR per batch
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
        
        # Compute RMSE from all predictions
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        mse = torch.mean((all_predictions - all_targets) ** 2)
        return torch.sqrt(mse).item()  # Return RMSE
    
    def _validate(self, val_loader: DataLoader, criterion) -> float:
        """Validate the model."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        mse = torch.mean((all_predictions - all_targets) ** 2)
        return torch.sqrt(mse).item()  # Return RMSE
    
    def test(
        self,
        test_loader: DataLoader,
        target_std: Optional[float] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            target_std: Target standard deviation for denormalization
            return_predictions: Whether to include predictions in results
            
        Returns:
            Dictionary with test metrics (and optionally predictions)
        """
        if self.is_pytorch:
            self.model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(self.device)
                    predictions = self.model(features)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
            
            predictions = torch.cat(all_predictions).numpy()
            targets = torch.cat(all_targets).numpy()
        else:
            # LightGBM model
            all_predictions = []
            all_targets = []
            
            for features, tgt in test_loader:
                X = features.numpy()
                pred = self.model.predict(X)
                all_predictions.append(pred.reshape(-1, 1))
                all_targets.append(tgt.numpy())
            
            predictions = np.concatenate(all_predictions)
            targets = np.concatenate(all_targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        results = {
            "rmse_normalized": float(rmse),
            "mae_normalized": float(mae),
        }
        
        # Denormalize if target_std provided
        if target_std is not None:
            results["rmse_denormalized"] = float(rmse * target_std)
            results["mae_denormalized"] = float(mae * target_std)
        
        # Include predictions if requested
        if return_predictions:
            results["predictions"] = predictions
            results["targets"] = targets
        
        return results
    
    def _train_sklearn_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        show_progress: bool,
        save_best: bool,
        verbose: int,
    ) -> Dict[str, Any]:
        """Train scikit-learn compatible models (e.g., LightGBM)."""
        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Training LightGBM Model")
            print(f"{'='*70}\n")
        
        # Extract data from loaders
        X_train, y_train = [], []
        for features, targets in train_loader:
            X_train.append(features.numpy())
            y_train.append(targets.numpy().ravel())
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        X_val, y_val = [], []
        for features, targets in val_loader:
            X_val.append(features.numpy())
            y_val.append(targets.numpy().ravel())
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        
        # Train model
        self.model.fit(X_train, y_train, X_val, y_val, verbose=(verbose >= 2))
        
        # Compute validation RMSE
        val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        
        self.history['train_loss'].append(0.0)  # Not tracked for LightGBM
        self.history['val_loss'].append(val_rmse)
        self.best_val_loss = val_rmse
        
        if verbose >= 1:
            print(f"Validation RMSE: {val_rmse:.6f}")
        
        # Save model
        if save_best and self.checkpoint_dir:
            self.model.save(str(self.checkpoint_dir / "best_model.pkl"))
        
        return self.history
    
    def _create_optimizer(self, name: str, lr: float, weight_decay: float):
        """Create optimizer."""
        if name.lower() == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name.lower() == "sgd":
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _create_criterion(self, name: str) -> nn.Module:
        """Create loss function."""
        if name.lower() == "mse":
            return nn.MSELoss()
        elif name.lower() == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {name}")
    
    def _create_scheduler(
        self,
        optimizer,
        name: str,
        num_epochs: int,
        steps_per_epoch: int,
    ):
        """Create learning rate scheduler."""
        if name.lower() == "none":
            return None
        elif name.lower() == "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        elif name.lower() == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.defaults['lr'],
                total_steps=num_epochs * steps_per_epoch,
                pct_start=0.3,
            )
        else:
            raise ValueError(f"Unknown scheduler: {name}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)


__all__ = ["Trainer"]
