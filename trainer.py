"""Training utilities for the surrogate model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime
import numpy as np
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ModelTrainer:
    """Trainer for the chemical density surrogate model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
    ):
        """Initialize the trainer.
        
        Args:
            model: PyTorch model to train or scikit-learn compatible model (e.g., LightGBM)
            device: Device to train on ('cuda' or 'cpu') - ignored for scikit-learn models
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if this is a PyTorch model or scikit-learn compatible model
        self.is_pytorch = isinstance(model, nn.Module)
        
        if self.is_pytorch:
            self.model.to(self.device)
        
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
        }
    
    def configure_optimizer(
        self,
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
    ):
        """Configure the optimizer.
        
        Args:
            optimizer_name: "adam" or "sgd"
            learning_rate: Learning rate
            weight_decay: L2 regularization strength (weight decay)
        
        Returns:
            Configured optimizer
        """
        if optimizer_name.lower() == "adam":
            return Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            return SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def configure_scheduler(
        self,
        optimizer,
        scheduler_name: str = "none",
        num_epochs: int = 100,
        steps_per_epoch: int = 100,
        onecycle_pct_start: float = 0.3,
        onecycle_anneal_strategy: str = "cos",
        cosine_t_max: Optional[int] = None,
        cosine_eta_min: float = 1e-6,
    ):
        """Configure the learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            scheduler_name: "none", "onecycle", or "cosine"
            num_epochs: Number of epochs
            steps_per_epoch: Steps per epoch (for onecycle)
            onecycle_pct_start: Percentage of cycle for increasing LR
            onecycle_anneal_strategy: "cos" or "linear"
            cosine_t_max: Max iterations for cosine annealing
            cosine_eta_min: Minimum learning rate for cosine
        
        Returns:
            Scheduler or None
        """
        if scheduler_name.lower() == "none":
            return None
        elif scheduler_name.lower() == "onecycle":
            total_steps = num_epochs * steps_per_epoch
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.defaults['lr'],
                total_steps=total_steps,
                pct_start=onecycle_pct_start,
                anneal_strategy=onecycle_anneal_strategy,
            )
        elif scheduler_name.lower() == "cosine":
            if cosine_t_max is None:
                cosine_t_max = num_epochs
            return CosineAnnealingLR(
                optimizer,
                T_max=cosine_t_max,
                eta_min=cosine_eta_min,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def configure_loss(self, loss_name: str = "mse") -> nn.Module:
        """Configure the loss function.
        
        Args:
            loss_name: "mse" or "mae"
        
        Returns:
            Loss function
        """
        if loss_name.lower() == "mse":
            return nn.MSELoss()
        elif loss_name.lower() == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        criterion,
        scheduler=None,
        show_progress: bool = True,
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Optional learning rate scheduler
            show_progress: Whether to show progress indicator
        
        Returns:
            Average training RMSE
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(train_loader, disable=not (show_progress and HAS_TQDM)) if HAS_TQDM else train_loader
        
        for features, targets in iterator:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(features)
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update scheduler step if it's OneCycleLR
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            if HAS_TQDM and show_progress:
                current_rmse = torch.sqrt(torch.tensor(total_loss / num_batches)).item()
                iterator.set_postfix({'RMSE': f'{current_rmse:.6f}'})
        
        # Return RMSE
        avg_mse = total_loss / num_batches
        return torch.sqrt(torch.tensor(avg_mse)).item()
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion,
    ) -> float:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Validation RMSE computed from all residuals (correct calculation)
        """
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
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute RMSE correctly: sqrt(mean(residuals²))
        # This is mathematically equivalent to sqrt(mean((pred - target)²))
        mse = torch.mean((all_predictions - all_targets) ** 2)
        rmse = torch.sqrt(mse).item()
        return rmse
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        loss_name: str = "mse",
        weight_decay: float = 0.0,
        scheduler_name: str = "none",
        onecycle_pct_start: float = 0.3,
        cosine_t_max: int = None,
        cosine_eta_min: float = 1e-6,
        show_progress_bar: bool = True,
        save_best_model: bool = True,
    ) -> Dict:
        """Complete training loop with learning rate scheduling and regularization.
        
        Handles both PyTorch and scikit-learn compatible models (e.g., LightGBM).
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Initial learning rate
            optimizer_name: Optimizer name ("adam" or "sgd")
            loss_name: Loss function name ("mse")
            weight_decay: L2 regularization strength
            scheduler_name: Learning rate scheduler ("none", "onecycle", or "cosine")
            onecycle_pct_start: OneCycleLR - percent of cycle spent increasing LR
            cosine_t_max: CosineAnnealingLR - max iterations (None = num_epochs)
            cosine_eta_min: CosineAnnealingLR - minimum learning rate
            show_progress_bar: Whether to show batch progress bar
            save_best_model: Whether to save the best model
        
        Returns:
            Training history dictionary
        """
        # Handle LightGBM models separately
        if not self.is_pytorch:
            return self._train_sklearn_model(
                train_loader, val_loader, num_epochs, show_progress_bar, save_best_model
            )
        
        # PyTorch training logic
        optimizer = self.configure_optimizer(optimizer_name, learning_rate, weight_decay)
        criterion = self.configure_loss(loss_name)
        
        # Set cosine_t_max if not specified
        if cosine_t_max is None:
            cosine_t_max = num_epochs
        
        # Configure learning rate scheduler
        scheduler = self.configure_scheduler(
            scheduler_name=scheduler_name,
            optimizer=optimizer,
            num_epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            onecycle_pct_start=onecycle_pct_start,
            cosine_t_max=cosine_t_max,
            cosine_eta_min=cosine_eta_min,
        )
        
        print(f"\n{'=' * 70}")
        print(f"Starting training on {self.device}")
        print(f"{'=' * 70}")
        print(f"Optimizer: {optimizer_name} (lr={learning_rate})")
        print(f"Loss: {loss_name}")
        print(f"Scheduler: {scheduler_name}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"{'=' * 70}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_rmse = self.train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler=scheduler,
                show_progress=show_progress_bar,
            )
            
            # Validate
            val_rmse = self.validate(val_loader, criterion)
            
            # Step scheduler (for Cosine and others that update per epoch)
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['train_loss'].append(train_rmse)
            self.training_history['val_loss'].append(val_rmse)
            self.training_history['epochs'].append(epoch + 1)
            
            # Print epoch results
            log_str = (f"Epoch {epoch + 1:3d}/{num_epochs} - "
                      f"Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f} | "
                      f"LR: {current_lr:.2e}")
            print(log_str, end="")
            
            # Save best model
            if save_best_model and val_rmse < self.best_val_loss:
                self.best_val_loss = val_rmse
                self.save_checkpoint(epoch + 1, is_best=True)
                print(" * (Best)", end="")  # Changed from checkmark to *
            
            print()
        
        print(f"\n{'=' * 70}")
        print(f"Training completed!")
        print(f"Best validation RMSE: {self.best_val_loss:.6f}")
        print(f"{'=' * 70}\n")
        
        return self.training_history
    
    def _train_sklearn_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        show_progress_bar: bool = True,
        save_best_model: bool = True,
    ) -> Dict:
        """Train scikit-learn compatible models like LightGBM.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (iterations)
            show_progress_bar: Whether to show progress
            save_best_model: Whether to save the best model
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'=' * 70}")
        print(f"Starting LightGBM training")
        print(f"{'=' * 70}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"{'=' * 70}\n")
        
        # Extract all data from loaders
        X_train = []
        y_train = []
        for features, targets in train_loader:
            X_train.append(features.numpy())
            y_train.append(targets.numpy().ravel())
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        
        X_val = []
        y_val = []
        for features, targets in val_loader:
            X_val.append(features.numpy())
            y_val.append(targets.numpy().ravel())
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        
        # Train the model
        self.model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=show_progress_bar,
        )
        
        # Get best validation RMSE from LightGBM's internal training history
        # LightGBM stores validation scores in evals_result_
        best_val_rmse = None
        if hasattr(self.model, 'evals_result_'):
            # evals_result_ is a dict like {'valid_0': {'rmse': [...]}}
            val_scores = self.model.evals_result_.get('valid_0', {}).get('rmse', [])
            if val_scores:
                best_val_rmse = float(min(val_scores))  # Best (lowest) validation RMSE
        
        # If we can't get best from history, use best_iteration
        if best_val_rmse is None and hasattr(self.model, 'best_iteration'):
            best_iteration = self.model.best_iteration
            val_pred = self.model.predict(X_val, num_iteration=best_iteration)
            best_val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        
        # Fallback: use final validation RMSE (not ideal but better than nothing)
        if best_val_rmse is None:
            val_pred = self.model.predict(X_val)
            best_val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        
        # Calculate final validation and training RMSE for logging
        val_pred_final = self.model.predict(X_val)
        val_rmse_final = np.sqrt(np.mean((val_pred_final - y_val) ** 2))
        train_rmse = np.sqrt(np.mean((self.model.predict(X_train) - y_train) ** 2))
        
        self.best_val_loss = best_val_rmse
        # Convert numpy values to Python floats for JSON serialization
        self.training_history['train_loss'].append(float(train_rmse))
        self.training_history['val_loss'].append(best_val_rmse)
        self.training_history['epochs'].append(1)  # LightGBM handles its own epochs
        
        # Save model
        if save_best_model:
            self.save_checkpoint(1, is_best=True)
        
        print(f"\n{'=' * 70}")
        print(f"Training completed!")
        print(f"Best validation RMSE: {self.best_val_loss:.6f}")
        print(f"{'=' * 70}\n")
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        filepath = self.checkpoint_dir / filename
        
        if self.is_pytorch:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model.config.__dict__ if hasattr(self.model, 'config') else None,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(checkpoint, filepath)
        else:
            # For scikit-learn models (e.g., LightGBM), use pickle
            self.model.save(str(filepath))
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load a model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        if self.is_pytorch:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {filepath}")
        else:
            # For scikit-learn models
            from lightgbm_model import LightGBMSurrogate
            self.model = LightGBMSurrogate.load(filepath)
            print(f"Loaded checkpoint from {filepath}")
    
    def test(
        self,
        test_loader: DataLoader,
        criterion,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluate on test set.
        
        Handles both PyTorch and scikit-learn compatible models.
        
        Args:
            test_loader: Test data loader
            criterion: Loss function (ignored for scikit-learn models)
        
        Returns:
            Tuple of (test_rmse, predictions, targets)
        """
        # Handle LightGBM models
        if not self.is_pytorch:
            return self._test_sklearn_model(test_loader)
        
        # PyTorch test logic
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute RMSE correctly: sqrt(mean(residuals²))
        # NOT batch-averaged: total_loss / len(test_loader) is incorrect
        mse = torch.mean((all_predictions - all_targets) ** 2)
        test_rmse = torch.sqrt(mse).item()
        
        return test_rmse, all_predictions, all_targets
    
    def _test_sklearn_model(
        self,
        test_loader: DataLoader,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Test scikit-learn compatible models like LightGBM.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Tuple of (test_rmse, predictions, targets) as PyTorch tensors
        """
        # Extract all test data
        X_test = []
        y_test = []
        for features, targets in test_loader:
            X_test.append(features.numpy())
            y_test.append(targets.numpy().ravel())
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Convert to torch tensors for compatibility
        predictions_tensor = torch.from_numpy(predictions.reshape(-1, 1)).float()
        targets_tensor = torch.from_numpy(y_test.reshape(-1, 1)).float()
        
        return rmse, predictions_tensor, targets_tensor
