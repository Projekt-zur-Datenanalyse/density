"""
Data loading and preprocessing utilities.

This module provides a unified data loading interface that:
- Loads the chemical density dataset from CSV
- Handles train/val/test splitting with reproducible seeds
- Provides normalization with proper train-only statistics
- Creates PyTorch DataLoader objects for training
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class Dataset(TorchDataset):
    """PyTorch Dataset for chemical density prediction.
    
    Features: SigC, SigH, EpsC, EpsH (Lennard-Jones parameters)
    Target: Density (kg/m³)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
    ):
        """Initialize the dataset.
        
        Args:
            features: Input features of shape (N, 4)
            targets: Target densities of shape (N,)
            normalize_features: Whether to normalize features
            normalize_targets: Whether to normalize targets
            feature_mean: Pre-computed feature means (from training set)
            feature_std: Pre-computed feature stds (from training set)
            target_mean: Pre-computed target mean (from training set)
            target_std: Pre-computed target std (from training set)
        """
        # Store original data
        self.features_raw = features.copy()
        self.targets_raw = targets.copy()
        
        # Normalization settings
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        
        # Compute or use provided statistics
        if normalize_features:
            if feature_mean is None or feature_std is None:
                self.feature_mean = features.mean(axis=0)
                self.feature_std = features.std(axis=0)
                self.feature_std[self.feature_std == 0] = 1.0
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
        else:
            self.feature_mean = None
            self.feature_std = None
        
        if normalize_targets:
            if target_mean is None or target_std is None:
                self.target_mean = float(targets.mean())
                self.target_std = float(targets.std())
                if self.target_std == 0:
                    self.target_std = 1.0
            else:
                self.target_mean = target_mean
                self.target_std = target_std
        else:
            self.target_mean = None
            self.target_std = None
        
        # Apply normalization
        if normalize_features:
            features = (features - self.feature_mean) / self.feature_std
        if normalize_targets:
            targets = (targets - self.target_mean) / self.target_std
        
        # Convert to tensors
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
    
    def denormalize_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize target predictions."""
        if not self.normalize_targets:
            return normalized
        return normalized * self.target_std + self.target_mean
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        return {
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }


class DataLoader:
    """Utility class for loading and splitting chemical density data."""
    
    FEATURE_COLUMNS = ["SigC", "SigH", "EpsC", "EpsH"]
    TARGET_COLUMN = "density"
    
    def __init__(self, data_path: str = "dataset.csv"):
        """Initialize the data loader.
        
        Args:
            data_path: Path to the dataset CSV file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    def load(
        self,
        validation_split: float = 0.15,
        test_split: float = 0.10,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 46,
        normalize: bool = True,
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, Dict[str, Any]]:
        """Load and split the dataset.
        
        IMPORTANT: Normalization statistics are computed ONLY on the training set
        to prevent data leakage.
        
        Args:
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing  
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            seed: Random seed for reproducible splits
            normalize: Whether to normalize features and targets
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, stats_dict)
            stats_dict contains normalization statistics and data info
        """
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Extract features and targets
        features = df[self.FEATURE_COLUMNS].values.astype(np.float32)
        targets = df[self.TARGET_COLUMN].values.astype(np.float32)
        
        total_samples = len(features)
        
        # Calculate split sizes
        test_size = int(test_split * total_samples)
        val_size = int(validation_split * total_samples)
        train_size = total_samples - val_size - test_size
        
        # Shuffle indices reproducibly
        np.random.seed(seed)
        indices = np.random.permutation(total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Split data
        X_train, y_train = features[train_indices], targets[train_indices]
        X_val, y_val = features[val_indices], targets[val_indices]
        X_test, y_test = features[test_indices], targets[test_indices]
        
        # Compute normalization stats from TRAINING SET ONLY
        if normalize:
            feature_mean = X_train.mean(axis=0)
            feature_std = X_train.std(axis=0)
            feature_std[feature_std == 0] = 1.0
            target_mean = float(y_train.mean())
            target_std = float(y_train.std())
            if target_std == 0:
                target_std = 1.0
        else:
            feature_mean = feature_std = None
            target_mean = target_std = None
        
        # Create datasets with same normalization stats
        train_dataset = Dataset(
            X_train, y_train, normalize, normalize,
            feature_mean, feature_std, target_mean, target_std
        )
        val_dataset = Dataset(
            X_val, y_val, normalize, normalize,
            feature_mean, feature_std, target_mean, target_std
        )
        test_dataset = Dataset(
            X_test, y_test, normalize, normalize,
            feature_mean, feature_std, target_mean, target_std
        )
        
        # Create data loaders
        train_loader = TorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = TorchDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = TorchDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Compile statistics
        stats = {
            "total_samples": total_samples,
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
            "feature_names": self.FEATURE_COLUMNS,
            "target_name": self.TARGET_COLUMN,
            "normalization": train_dataset.get_normalization_stats(),
            "feature_ranges": {
                name: {"min": float(features[:, i].min()), "max": float(features[:, i].max())}
                for i, name in enumerate(self.FEATURE_COLUMNS)
            },
            "target_range": {"min": float(targets.min()), "max": float(targets.max())},
            "target_std_original": float(targets.std()),
        }
        
        return train_loader, val_loader, test_loader, stats
    
    def load_with_fixed_test_split(
        self,
        master_seed: int,
        train_val_seed: int,
        train_ratio: float = 0.75,
        val_ratio: float = 0.20,
        test_ratio: float = 0.05,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load data with fixed test split (for ensemble training).
        
        Uses master_seed to determine test indices (fixed across all models),
        then uses train_val_seed to shuffle train/val split differently per model.
        
        Args:
            master_seed: Seed for fixed test split selection
            train_val_seed: Seed for train/val split variation
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            batch_size: Batch size for data loaders
            normalize: Whether to normalize data
        
        Returns:
            train_loader, val_loader, test_loader, X_test, y_test, norm_stats
        """
        # Load CSV
        df = pd.read_csv(self.data_path)
        features = df[self.FEATURE_COLUMNS].values.astype(np.float32)
        targets = df[self.TARGET_COLUMN].values.astype(np.float32)
        
        total_samples = len(features)
        
        # Use master seed to determine test indices (FIXED across all models)
        np.random.seed(master_seed)
        all_indices = np.random.permutation(total_samples)
        
        n_train_val = int(total_samples * (train_ratio + val_ratio))
        train_val_indices = all_indices[:n_train_val]
        test_indices = all_indices[n_train_val:]
        
        # Use train_val_seed to shuffle train/val split (VARIABLE per model)
        np.random.seed(train_val_seed)
        train_val_shuffled = np.random.permutation(train_val_indices)
        
        n_train = int(len(train_val_shuffled) * (train_ratio / (train_ratio + val_ratio)))
        train_indices = train_val_shuffled[:n_train]
        val_indices = train_val_shuffled[n_train:]
        
        # Extract data
        X_train, y_train = features[train_indices], targets[train_indices]
        X_val, y_val = features[val_indices], targets[val_indices]
        X_test, y_test = features[test_indices], targets[test_indices]
        
        # Compute normalization stats from training set
        if normalize:
            feature_mean = X_train.mean(axis=0)
            feature_std = X_train.std(axis=0)
            feature_std[feature_std == 0] = 1.0
            target_mean = float(y_train.mean())
            target_std = float(y_train.std())
            if target_std == 0:
                target_std = 1.0
        else:
            feature_mean = feature_std = None
            target_mean = target_std = None
        
        # Create datasets
        train_dataset = Dataset(
            X_train, y_train, normalize, normalize,
            feature_mean, feature_std, target_mean, target_std
        )
        val_dataset = Dataset(
            X_val, y_val, normalize, normalize,
            feature_mean, feature_std, target_mean, target_std
        )
        test_dataset = Dataset(
            X_test, y_test, normalize, normalize,
            feature_mean, feature_std, target_mean, target_std
        )
        
        # Create loaders
        train_loader = TorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = TorchDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = TorchDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        norm_stats = {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }
        
        return train_loader, val_loader, test_loader, X_test, y_test, norm_stats


__all__ = ["DataLoader", "Dataset"]
