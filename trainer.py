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
            model: PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
            Average validation RMSE
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Return RMSE
        avg_mse = total_loss / num_batches
        return torch.sqrt(torch.tensor(avg_mse)).item()
    
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config.__dict__ if hasattr(self.model, 'config') else None,
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load a model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
    
    def test(
        self,
        test_loader: DataLoader,
        criterion,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            criterion: Loss function
        
        Returns:
            Tuple of (test_rmse, predictions, targets)
        """
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
        avg_mse = total_loss / len(test_loader)
        avg_rmse = torch.sqrt(torch.tensor(avg_mse)).item()
        
        return avg_rmse, all_predictions, all_targets
