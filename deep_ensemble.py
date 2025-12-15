"""
Deep Ensemble Training for Chemical Density Prediction.

This module implements deep ensemble modeling for improved predictions and uncertainty quantification.
Key features:
- Train X models of the same architecture (same-architecture ensemble)
- Train 1 model per selected architecture (multi-architecture ensemble)
- Fixed test split across all models (critical for fair evaluation)
- Different train/val splits per model (different priors)
- Mean prediction as final output
- Uncertainty quantification via prediction standard deviation
- Comprehensive visualization of results
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import local modules
from config import ModelConfig
from data_loader import ChemicalDensityDataLoader
from trainer import ModelTrainer
from model import ChemicalDensitySurrogate

# Optional imports
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

# Visualization
import matplotlib.pyplot as plt

# Available architectures
ARCHITECTURES = ["mlp", "cnn", "cnn_multiscale", "lightgbm"]


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(architecture: str, model_config: ModelConfig):
    """Create a model based on architecture type.
    
    Args:
        architecture: One of 'mlp', 'cnn', 'cnn_multiscale', 'lightgbm'
        model_config: Configuration object
    
    Returns:
        Model instance
    """
    if architecture == "mlp":
        return ChemicalDensitySurrogate(model_config)
    
    elif architecture == "cnn":
        if not CNN_AVAILABLE:
            raise ImportError("CNN requires cnn_model.py to be available")
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
            raise ImportError("CNN Multi-Scale requires cnn_model.py to be available")
        return MultiScaleConvolutionalSurrogate(
            num_input_features=model_config.input_dim,
            expansion_size=model_config.cnn_multiscale_expansion_size,
            num_scales=model_config.cnn_multiscale_num_scales,
            base_channels=model_config.cnn_multiscale_base_channels,
            dropout_rate=model_config.dropout_rate,
        )
    
    elif architecture == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM requires lightgbm package. Install with: pip install lightgbm")
        return LightGBMSurrogate(
            num_leaves=model_config.lgb_num_leaves,
            learning_rate=model_config.lgb_learning_rate,
            num_boost_round=model_config.lgb_num_boost_round,
            max_depth=model_config.lgb_max_depth,
            min_child_samples=model_config.lgb_min_child_samples,
            subsample=model_config.lgb_subsample,
            colsample_bytree=model_config.lgb_colsample_bytree,
            reg_alpha=model_config.lgb_reg_alpha,
            reg_lambda=model_config.lgb_reg_lambda,
            boosting_type=model_config.lgb_boosting_type,
            metric=model_config.lgb_metric,
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_data_with_fixed_test_split(
    data_dir: str,
    master_seed: int,
    train_val_seed: int,
    train_ratio: float = 0.75,
    val_ratio: float = 0.20,
    test_ratio: float = 0.05,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, Dict]:
    """Load data with fixed test split and variable train/val split.
    
    Uses master_seed to determine test indices (fixed across all models),
    then uses train_val_seed to shuffle train/val split differently per model.
    
    Args:
        data_dir: Directory containing the dataset
        master_seed: Seed for fixed test split selection
        train_val_seed: Seed for train/val split variation
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        batch_size: Batch size for data loaders
    
    Returns:
        train_loader, val_loader, test_loader, X_test, y_test, norm_stats
    """
    # Load the raw data using the data_loader module's unified dataset loading
    from data_loader import USE_UNIFIED_DATASET
    
    data_path = Path(data_dir)
    
    if USE_UNIFIED_DATASET:
        # Load from unified Dataset.csv
        csv_path = data_path / "Dataset.csv"
        df = pd.read_csv(csv_path)
        all_features = df[['SigC', 'SigH', 'EpsC', 'EpsH']].values.astype(np.float32)
        all_targets = df['density'].values.astype(np.float32)
    else:
        # Load from multiple Dataset_*.csv files
        feature_cols = []
        target_cols = []
        for csv_file in sorted(data_path.glob("Dataset_*.csv")):
            df = pd.read_csv(csv_file)
            feature_cols.append(df[['SigC', 'SigH', 'EpsC', 'EpsH']].values)
            target_cols.append(df['density'].values)
        all_features = np.concatenate(feature_cols, axis=0).astype(np.float32)
        all_targets = np.concatenate(target_cols, axis=0).astype(np.float32)
    
    total_samples = len(all_features)
    
    # Use master seed to determine test indices (FIXED across all models)
    np.random.seed(master_seed)
    all_indices = np.random.permutation(total_samples)
    
    # Calculate split sizes
    n_train_val = int(total_samples * (train_ratio + val_ratio))
    
    # Split into train+val and test (test is FIXED)
    train_val_indices = all_indices[:n_train_val]
    test_indices = all_indices[n_train_val:]
    
    # Now use train_val_seed to shuffle train/val split (VARIABLE per model)
    np.random.seed(train_val_seed)
    train_val_shuffled = np.random.permutation(train_val_indices)
    
    n_train = int(len(train_val_shuffled) * (train_ratio / (train_ratio + val_ratio)))
    train_indices = train_val_shuffled[:n_train]
    val_indices = train_val_shuffled[n_train:]
    
    # Extract data using indices
    X_train = all_features[train_indices]
    y_train = all_targets[train_indices]
    X_val = all_features[val_indices]
    y_val = all_targets[val_indices]
    X_test = all_features[test_indices]
    y_test = all_targets[test_indices]
    
    # Compute normalization stats from training set ONLY
    feature_mean = np.mean(X_train, axis=0)
    feature_std = np.std(X_train, axis=0)
    feature_std[feature_std == 0] = 1.0  # Avoid division by zero
    
    target_mean = np.mean(y_train)
    target_std = np.std(y_train)
    if target_std == 0:
        target_std = 1.0
    
    # Normalize all splits
    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std
    
    y_train_norm = (y_train - target_mean) / target_std
    y_val_norm = (y_val - target_mean) / target_std
    y_test_norm = (y_test - target_mean) / target_std
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train_norm).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val_norm).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test_norm).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Store normalization stats
    norm_stats = {
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': float(target_mean),
        'target_std': float(target_std),
    }
    
    return train_loader, val_loader, test_loader, X_test, y_test, norm_stats


class DeepEnsemble:
    """Deep Ensemble for uncertainty-aware predictions."""
    
    def __init__(
        self,
        output_dir: str = "./ensemble_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the deep ensemble.
        
        Args:
            output_dir: Directory to save results
            device: Device for PyTorch models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.models: List[Any] = []
        self.model_configs: List[Dict] = []
        self.normalization_stats: List[Dict] = []
        self.training_histories: List[Dict] = []
        
    def train_ensemble(
        self,
        architectures: List[str],
        model_config: ModelConfig,
        data_dir: str = ".",
        train_ratio: float = 0.75,
        val_ratio: float = 0.20,
        test_ratio: float = 0.05,
        batch_size: int = 32,
        num_models_per_arch: int = 1,
        master_seed: int = 46,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: str = "none",
        show_progress: bool = True,
    ) -> Dict:
        """Train the ensemble models.
        
        Args:
            architectures: List of architecture names to train
            model_config: Base model configuration
            data_dir: Directory containing the dataset
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            batch_size: Batch size for training
            num_models_per_arch: Number of models per architecture
            master_seed: Seed for fixed test split
            num_epochs: Training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
            optimizer: Optimizer name
            scheduler: Scheduler name
            show_progress: Show training progress
        
        Returns:
            Summary dictionary
        """
        total_models = len(architectures) * num_models_per_arch
        print(f"\n{'=' * 80}")
        print(f"DEEP ENSEMBLE TRAINING")
        print(f"{'=' * 80}")
        print(f"Architectures: {architectures}")
        print(f"Models per architecture: {num_models_per_arch}")
        print(f"Total models: {total_models}")
        print(f"Master seed (test split): {master_seed}")
        print(f"Device: {self.device}")
        print(f"{'=' * 80}\n")
        
        model_idx = 0
        for arch in architectures:
            for i in range(num_models_per_arch):
                model_idx += 1
                
                # Generate unique seed for this model's train/val split
                train_val_seed = master_seed + model_idx * 1000
                
                print(f"\n{'─' * 70}")
                print(f"Training Model {model_idx}/{total_models}: {arch.upper()} "
                      f"(seed={train_val_seed})")
                print(f"{'─' * 70}")
                
                # Set seeds for reproducibility
                set_all_seeds(train_val_seed)
                
                # Load data with fixed test split, variable train/val
                train_loader, val_loader, test_loader, X_test, y_test, norm_stats = \
                    load_data_with_fixed_test_split(
                        data_dir=data_dir,
                        master_seed=master_seed,
                        train_val_seed=train_val_seed,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        batch_size=batch_size,
                    )
                
                # Create model
                model_config.architecture = arch
                model = create_model(arch, model_config)
                
                # Create checkpoint directory for this model
                model_dir = self.output_dir / f"model_{model_idx}_{arch}"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Train model
                trainer = ModelTrainer(
                    model=model,
                    device=self.device,
                    checkpoint_dir=str(model_dir / "checkpoints"),
                )
                
                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    optimizer_name=optimizer,
                    scheduler_name=scheduler,
                    show_progress_bar=show_progress,
                    save_best_model=True,
                )
                
                # Store model and metadata
                self.models.append({
                    'model': model,
                    'architecture': arch,
                    'trainer': trainer,
                    'seed': train_val_seed,
                    'model_dir': str(model_dir),
                })
                self.normalization_stats.append(norm_stats)
                self.training_histories.append(history)
                
                # Save model config
                config_dict = {
                    'architecture': arch,
                    'seed': train_val_seed,
                    'model_idx': model_idx,
                    'best_val_loss': trainer.best_val_loss,
                }
                if hasattr(model, 'get_num_parameters'):
                    config_dict['num_parameters'] = model.get_num_parameters()
                elif hasattr(model, 'parameters'):
                    config_dict['num_parameters'] = sum(p.numel() for p in model.parameters())
                
                with open(model_dir / 'model_config.json', 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                self.model_configs.append(config_dict)
        
        # Store test data for predictions
        self._test_loader = test_loader
        self._X_test = X_test
        self._y_test = y_test
        self._master_seed = master_seed
        
        # Save ensemble summary
        summary = self._create_summary()
        with open(self.output_dir / 'ensemble_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make ensemble predictions with uncertainty.
        
        Returns:
            Tuple of (mean_predictions, std_predictions, individual_predictions)
        """
        all_predictions = []
        
        for model_info in self.models:
            model = model_info['model']
            norm_stats = self.normalization_stats[self.models.index(model_info)]
            
            # Get predictions
            if isinstance(model, nn.Module):
                model.eval()
                predictions = []
                with torch.no_grad():
                    for features, _ in self._test_loader:
                        features = features.to(self.device)
                        pred = model(features)
                        predictions.append(pred.cpu().numpy())
                predictions = np.concatenate(predictions, axis=0)
            else:
                # LightGBM or sklearn model
                X_test_norm = (self._X_test - norm_stats['feature_mean']) / norm_stats['feature_std']
                predictions = model.predict(X_test_norm).reshape(-1, 1)
            
            # Denormalize predictions
            predictions = predictions * norm_stats['target_std'] + norm_stats['target_mean']
            all_predictions.append(predictions.flatten())
        
        # Stack all predictions: shape (num_models, num_samples)
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Compute ensemble statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        return mean_predictions, std_predictions, all_predictions
    
    def evaluate(self) -> Dict:
        """Evaluate ensemble performance.
        
        Returns:
            Evaluation metrics dictionary
        """
        mean_preds, std_preds, all_preds = self.predict()
        
        # Get true targets
        true_targets = self._y_test.flatten()
        
        # Compute metrics for ensemble mean
        errors = mean_preds - true_targets
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        
        # Compute metrics for individual models
        individual_rmses = []
        for i, preds in enumerate(all_preds):
            ind_errors = preds - true_targets
            ind_rmse = np.sqrt(np.mean(ind_errors ** 2))
            individual_rmses.append(ind_rmse)
        
        # Uncertainty calibration: check if std correlates with error
        abs_errors = np.abs(errors)
        correlation = np.corrcoef(std_preds, abs_errors)[0, 1]
        
        metrics = {
            'ensemble_rmse': float(rmse),
            'ensemble_mae': float(mae),
            'mean_individual_rmse': float(np.mean(individual_rmses)),
            'std_individual_rmse': float(np.std(individual_rmses)),
            'best_individual_rmse': float(np.min(individual_rmses)),
            'worst_individual_rmse': float(np.max(individual_rmses)),
            'mean_uncertainty': float(np.mean(std_preds)),
            'uncertainty_error_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'num_models': len(self.models),
            'num_test_samples': len(true_targets),
        }
        
        print(f"\n{'=' * 70}")
        print("ENSEMBLE EVALUATION RESULTS")
        print(f"{'=' * 70}")
        print(f"Ensemble RMSE:         {rmse:.6f}")
        print(f"Ensemble MAE:          {mae:.6f}")
        print(f"Mean Individual RMSE:  {np.mean(individual_rmses):.6f}")
        print(f"Best Individual RMSE:  {np.min(individual_rmses):.6f}")
        print(f"Mean Uncertainty:      {np.mean(std_preds):.6f}")
        print(f"Uncertainty-Error Corr: {correlation:.4f}")
        print(f"{'=' * 70}\n")
        
        # Save metrics
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def visualize(self, save_plots: bool = True) -> None:
        """Create comprehensive visualizations of ensemble results.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        mean_preds, std_preds, all_preds = self.predict()
        true_targets = self._y_test.flatten()
        errors = mean_preds - true_targets
        abs_errors = np.abs(errors)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 14))
        
        # 1. Predictions vs True Values with Uncertainty
        ax1 = fig.add_subplot(2, 3, 1)
        sorted_idx = np.argsort(true_targets)
        ax1.fill_between(
            range(len(true_targets)),
            (mean_preds - 2*std_preds)[sorted_idx],
            (mean_preds + 2*std_preds)[sorted_idx],
            alpha=0.3, color='blue', label='±2σ'
        )
        ax1.fill_between(
            range(len(true_targets)),
            (mean_preds - std_preds)[sorted_idx],
            (mean_preds + std_preds)[sorted_idx],
            alpha=0.5, color='blue', label='±1σ'
        )
        ax1.plot(true_targets[sorted_idx], 'g-', linewidth=2, label='True')
        ax1.plot(mean_preds[sorted_idx], 'r--', linewidth=1.5, label='Predicted Mean')
        ax1.set_xlabel('Sample (sorted by true value)')
        ax1.set_ylabel('Density')
        ax1.set_title('Predictions vs True Values')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter Plot: Predicted vs True
        ax2 = fig.add_subplot(2, 3, 2)
        scatter = ax2.scatter(true_targets, mean_preds, c=std_preds, 
                              cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax2, label='Uncertainty (σ)')
        
        # Add perfect prediction line
        min_val = min(true_targets.min(), mean_preds.min())
        max_val = max(true_targets.max(), mean_preds.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax2.set_xlabel('True Density')
        ax2.set_ylabel('Predicted Density')
        ax2.set_title('Predicted vs True (colored by uncertainty)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error Distribution
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                    label=f'Mean Error: {np.mean(errors):.4f}')
        ax3.set_xlabel('Error (Predicted - True)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Uncertainty vs Error
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.scatter(std_preds, abs_errors, alpha=0.6, s=20, c='steelblue')
        
        # Fit and plot trend line
        z = np.polyfit(std_preds, abs_errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(std_preds.min(), std_preds.max(), 100)
        ax4.plot(x_line, p(x_line), 'r-', linewidth=2, 
                 label=f'Trend (corr={np.corrcoef(std_preds, abs_errors)[0,1]:.3f})')
        ax4.set_xlabel('Uncertainty (σ)')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Uncertainty vs Error (Calibration)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Individual Model Predictions
        ax5 = fig.add_subplot(2, 3, 5)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_preds)))
        for i, preds in enumerate(all_preds):
            model_info = self.models[i]
            arch = model_info['architecture']
            label = f"{arch}_{i+1}"
            rmse_i = np.sqrt(np.mean((preds - true_targets) ** 2))
            ax5.scatter(true_targets, preds, alpha=0.4, s=15, c=[colors[i]], 
                       label=f'{label} (RMSE={rmse_i:.4f})')
        ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        ax5.set_xlabel('True Density')
        ax5.set_ylabel('Predicted Density')
        ax5.set_title('Individual Model Predictions')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Model RMSE Comparison
        ax6 = fig.add_subplot(2, 3, 6)
        individual_rmses = [np.sqrt(np.mean((preds - true_targets) ** 2)) 
                           for preds in all_preds]
        ensemble_rmse = np.sqrt(np.mean(errors ** 2))
        
        model_labels = [f"{self.models[i]['architecture']}_{i+1}" 
                       for i in range(len(self.models))]
        model_labels.append('ENSEMBLE')
        all_rmses = individual_rmses + [ensemble_rmse]
        
        bars = ax6.bar(range(len(all_rmses)), all_rmses, 
                      color=['steelblue']*len(individual_rmses) + ['darkred'])
        ax6.set_xticks(range(len(all_rmses)))
        ax6.set_xticklabels(model_labels, rotation=45, ha='right')
        ax6.set_ylabel('RMSE')
        ax6.set_title('Individual vs Ensemble RMSE')
        ax6.axhline(y=ensemble_rmse, color='red', linestyle='--', alpha=0.7)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, rmse in zip(bars, all_rmses):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'ensemble_visualization.png', dpi=150, 
                       bbox_inches='tight')
            print(f"Saved visualization to {self.output_dir / 'ensemble_visualization.png'}")
        
        plt.close(fig)
        
        # Create additional detailed uncertainty plot
        self._plot_uncertainty_detail(mean_preds, std_preds, true_targets, save_plots)
    
    def _plot_uncertainty_detail(
        self, 
        mean_preds: np.ndarray, 
        std_preds: np.ndarray, 
        true_targets: np.ndarray,
        save_plots: bool
    ) -> None:
        """Create detailed uncertainty visualization.
        
        Args:
            mean_preds: Mean ensemble predictions
            std_preds: Standard deviation of predictions
            true_targets: True target values
            save_plots: Whether to save plots
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sort by true value for visualization
        sorted_idx = np.argsort(true_targets)
        sorted_true = true_targets[sorted_idx]
        sorted_pred = mean_preds[sorted_idx]
        sorted_std = std_preds[sorted_idx]
        sorted_error = np.abs(sorted_pred - sorted_true)
        
        # 1. Uncertainty vs True Value
        ax1 = axes[0]
        ax1.scatter(sorted_true, sorted_std, alpha=0.6, s=20, c='steelblue')
        ax1.set_xlabel('True Density')
        ax1.set_ylabel('Prediction Uncertainty (σ)')
        ax1.set_title('Uncertainty vs True Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Error and Uncertainty bands
        ax2 = axes[1]
        x = np.arange(len(sorted_true))
        ax2.fill_between(x, -2*sorted_std, 2*sorted_std, alpha=0.3, 
                        color='blue', label='±2σ band')
        ax2.fill_between(x, -sorted_std, sorted_std, alpha=0.5, 
                        color='blue', label='±1σ band')
        ax2.scatter(x, sorted_pred - sorted_true, s=10, c='red', alpha=0.7, label='Errors')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Sample Index (sorted by true value)')
        ax2.set_ylabel('Error (Predicted - True)')
        ax2.set_title('Errors with Uncertainty Bands')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Calibration: % of errors within σ bounds
        ax3 = axes[2]
        sigmas = np.linspace(0, 3, 31)
        coverage = []
        for s in sigmas:
            within_bounds = np.abs(sorted_pred - sorted_true) <= s * sorted_std
            coverage.append(np.mean(within_bounds) * 100)
        
        # Expected coverage for Gaussian (theoretical)
        from scipy import stats
        expected = [(stats.norm.cdf(s) - stats.norm.cdf(-s)) * 100 for s in sigmas]
        
        ax3.plot(sigmas, coverage, 'b-', linewidth=2, label='Observed')
        ax3.plot(sigmas, expected, 'r--', linewidth=2, label='Expected (Gaussian)')
        ax3.fill_between(sigmas, coverage, expected, alpha=0.3, color='gray')
        ax3.set_xlabel('Number of Standard Deviations')
        ax3.set_ylabel('% of Errors Within Bounds')
        ax3.set_title('Uncertainty Calibration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 3)
        ax3.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'uncertainty_analysis.png', dpi=150, 
                       bbox_inches='tight')
            print(f"Saved uncertainty analysis to {self.output_dir / 'uncertainty_analysis.png'}")
        
        plt.close(fig)
    
    def _create_summary(self) -> Dict:
        """Create ensemble training summary.
        
        Returns:
            Summary dictionary
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(self.models),
            'architectures': [m['architecture'] for m in self.models],
            'seeds': [m['seed'] for m in self.models],
            'master_seed': self._master_seed,
            'model_configs': self.model_configs,
            'best_val_losses': [h['val_loss'][-1] if h['val_loss'] else None 
                               for h in self.training_histories],
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Ensemble Training for Chemical Density Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 5 MLP models (same architecture ensemble)
  python deep_ensemble.py --architectures mlp --num-models 5

  # Train multi-architecture ensemble (1 model each)
  python deep_ensemble.py --architectures mlp cnn lightgbm
  
  # Train 3 CNN + 3 MLP models
  python deep_ensemble.py --architectures cnn mlp --num-models 3
  
  # Use custom seeds and epochs
  python deep_ensemble.py --architectures mlp --num-models 8 --master-seed 42 --epochs 200
        """
    )
    
    # Architecture and ensemble settings
    parser.add_argument(
        "--architectures", "-a",
        nargs="+",
        choices=ARCHITECTURES,
        default=["mlp"],
        help="Architectures to include in ensemble (default: mlp)"
    )
    parser.add_argument(
        "--num-models", "-n",
        type=int,
        default=5,
        help="Number of models per architecture (default: 5)"
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=46,
        help="Master seed for fixed test split (default: 46)"
    )
    
    # Training settings
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay", "--wd",
        type=float,
        default=0.0,
        help="Weight decay / L2 regularization (default: 0.0)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer (default: adam)"
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "onecycle", "cosine"],
        default="none",
        help="LR scheduler (default: none)"
    )
    
    # Data settings
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.75,
        help="Training split ratio (default: 0.75)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Validation split ratio (default: 0.20)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.05,
        help="Test split ratio (default: 0.05)"
    )
    
    # Model architecture settings (MLP)
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=4,
        help="MLP expansion factor (default: 4)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./ensemble_results",
        help="Output directory (default: ./ensemble_results)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing the dataset (default: .)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_str = "_".join(args.architectures)
    output_dir = Path(args.output_dir) / f"ensemble_{arch_str}_{timestamp}"
    
    # Create model config
    model_config = ModelConfig(
        seed=args.master_seed,
        expansion_factor=args.expansion_factor,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
    )
    
    # Create and train ensemble
    ensemble = DeepEnsemble(
        output_dir=str(output_dir),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    summary = ensemble.train_ensemble(
        architectures=args.architectures,
        model_config=model_config,
        data_dir=args.data_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        batch_size=args.batch_size,
        num_models_per_arch=args.num_models,
        master_seed=args.master_seed,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        show_progress=not args.quiet,
    )
    
    # Evaluate ensemble
    metrics = ensemble.evaluate()
    
    # Visualize results
    if not args.no_visualize:
        ensemble.visualize(save_plots=True)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Ensemble RMSE: {metrics['ensemble_rmse']:.6f}")
    print(f"Best Individual RMSE: {metrics['best_individual_rmse']:.6f}")
    print(f"Improvement: {((metrics['mean_individual_rmse'] - metrics['ensemble_rmse']) / metrics['mean_individual_rmse'] * 100):.2f}%")


if __name__ == "__main__":
    main()
