"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional
from pathlib import Path
import os


# ============================================================================
# CONFIGURATION: Set this to True to load from unified Dataset.csv instead
#                of the 4 separate Dataset_*.csv files
# ============================================================================
USE_UNIFIED_DATASET = True  # Set to True to use Dataset.csv
# ============================================================================


class ChemicalDensityDataset(Dataset):
    """PyTorch Dataset for chemical density prediction.
    
    Loads data from CSV files with 5 rows:
    - Row 0: SigC (sigma carbon)
    - Row 1: SigH (sigma hydrogen)
    - Row 2: EpsC (epsilon carbon)
    - Row 3: EpsH (epsilon hydrogen)
    - Row 4: Density (target value)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        normalize_features: bool = False,
        normalize_targets: bool = False,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
    ):
        """Initialize the dataset.
        
        Args:
            features: Input features of shape (N, 4) - [SigC, SigH, EpsC, EpsH]
            targets: Target densities of shape (N,)
            normalize_features: Whether to normalize features
            normalize_targets: Whether to normalize targets
            feature_mean: Pre-computed feature means for normalization
            feature_std: Pre-computed feature standard deviations
            target_mean: Pre-computed target mean
            target_std: Pre-computed target standard deviation
        """
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)
        
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        
        # Store normalization statistics
        if normalize_features:
            if feature_mean is None or feature_std is None:
                self.feature_mean = features.mean(axis=0)
                self.feature_std = features.std(axis=0)
                # Avoid division by zero
                self.feature_std[self.feature_std == 0] = 1.0
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            
            # Normalize features
            self.features = (self.features - torch.from_numpy(self.feature_mean).float()) / \
                           torch.from_numpy(self.feature_std).float()
        else:
            self.feature_mean = None
            self.feature_std = None
        
        if normalize_targets:
            if target_mean is None or target_std is None:
                self.target_mean = targets.mean()
                self.target_std = targets.std()
                # Avoid division by zero
                if self.target_std == 0:
                    self.target_std = 1.0
            else:
                self.target_mean = target_mean
                self.target_std = target_std
            
            # Normalize targets
            self.targets = (self.targets - self.target_mean) / self.target_std
        else:
            self.target_mean = None
            self.target_std = None
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]
    
    def denormalize_targets(self, normalized_targets: np.ndarray) -> np.ndarray:
        """Denormalize target predictions.
        
        Args:
            normalized_targets: Normalized predictions
            
        Returns:
            Denormalized predictions
        """
        if not self.normalize_targets:
            return normalized_targets
        return normalized_targets * self.target_std + self.target_mean


class ChemicalDensityDataLoader:
    """Utility class for loading and splitting chemical density data."""
    
    def __init__(self, data_dir: str = "."):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
    
    def load_dataset(
        self,
        dataset_paths: Optional[List[str]] = None,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 46,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, ChemicalDensityDataset]:
        """Load and split the dataset.
        
        Args:
            dataset_paths: List of CSV file paths. If None, uses USE_UNIFIED_DATASET to decide
            normalize_features: Whether to normalize input features
            normalize_targets: Whether to normalize target values
            validation_split: Fraction of data to use for validation (0.0-1.0)
            test_split: Fraction of data to use for testing (0.0-1.0)
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            seed: Random seed for reproducible train/val/test splits (default: 46)
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, full_dataset)
        """
        # Determine which dataset format to use
        if USE_UNIFIED_DATASET:
            all_features, all_targets = self._load_unified_dataset()
        else:
            # Auto-discover dataset files if not provided
            if dataset_paths is None:
                dataset_paths = sorted(self.data_dir.glob("Dataset_*.csv"))
                dataset_paths = [str(p) for p in dataset_paths]
                print(f"Auto-discovered {len(dataset_paths)} dataset files")
            
            # Load all data
            all_features = []
            all_targets = []
            
            for dataset_path in dataset_paths:
                features, targets = self._load_single_dataset(dataset_path)
                all_features.append(features)
                all_targets.append(targets)
                print(f"Loaded {len(features)} samples from {Path(dataset_path).name}")
            
            # Combine all datasets
            all_features = np.concatenate(all_features, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        
        print(f"Total samples: {len(all_features)}")
        print(f"Features shape: {all_features.shape}")
        print(f"Targets shape: {all_targets.shape}")
        print(f"Feature ranges:")
        for i, name in enumerate(['SigC', 'SigH', 'EpsC', 'EpsH']):
            print(f"  {name}: [{all_features[:, i].min():.4f}, {all_features[:, i].max():.4f}]")
        print(f"Target (Density) range: [{all_targets.min():.2f}, {all_targets.max():.2f}]")
        print(f"Target std dev: {all_targets.std():.2f} kg/m³")
        
        # Calculate split sizes
        total_size = len(all_features)
        test_size = int(test_split * total_size)
        val_size = int(validation_split * (total_size - test_size))
        train_size = total_size - val_size - test_size
        
        print(f"Split: Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Set seed for reproducible random split
        np.random.seed(seed)
        
        # Split indices BEFORE creating datasets (to compute stats only on training set)
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Extract training data ONLY for computing normalization statistics
        train_features = all_features[train_indices]
        train_targets = all_targets[train_indices]
        
        # Compute normalization statistics ONLY on training set (prevents data leakage)
        if normalize_features:
            feature_mean = train_features.mean(axis=0)
            feature_std = train_features.std(axis=0)
            feature_std[feature_std == 0] = 1.0  # Avoid division by zero
        else:
            feature_mean = None
            feature_std = None
        
        if normalize_targets:
            target_mean = train_targets.mean()
            target_std = train_targets.std()
            if target_std == 0:
                target_std = 1.0
        else:
            target_mean = None
            target_std = None
        
        # Create datasets with the SAME normalization stats for all splits
        train_dataset = ChemicalDensityDataset(
            features=train_features,
            targets=train_targets,
            normalize_features=normalize_features,
            normalize_targets=normalize_targets,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        
        val_dataset = ChemicalDensityDataset(
            features=all_features[val_indices],
            targets=all_targets[val_indices],
            normalize_features=normalize_features,
            normalize_targets=normalize_targets,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        
        test_dataset = ChemicalDensityDataset(
            features=all_features[test_indices],
            targets=all_targets[test_indices],
            normalize_features=normalize_features,
            normalize_targets=normalize_targets,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        
        # Store the full dataset for reference (for getting target_std)
        full_dataset = train_dataset  # Return train dataset as full_dataset for backward compatibility
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        return train_loader, val_loader, test_loader, full_dataset
    
    @staticmethod
    def _load_single_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single dataset CSV file.
        
        Expected format (with first column as index):
        - Row 0: SigC values (first column has label "SigC", data from column 1 onwards)
        - Row 1: SigH values (first column has label "SigH", data from column 1 onwards)
        - Row 2: EpsC values (first column has label "EpsC", data from column 1 onwards)
        - Row 3: EpsH values (first column has label "EpsH", data from column 1 onwards)
        - Row 4: Density values (first column has label "density", data from column 1 onwards)
        
        Args:
            dataset_path: Path to the CSV file
        
        Returns:
            Tuple of (features, targets) arrays of shape (n_samples, 4) and (n_samples,)
        """
        # Read CSV with first column as index
        df = pd.read_csv(dataset_path, index_col=0)
        
        # Extract feature values (skip the first column which contains labels)
        # Row indices 0-3 are the features, row 4 is density
        sigc_values = df.loc[0].values[1:].astype(np.float32)
        sigh_values = df.loc[1].values[1:].astype(np.float32)
        epsc_values = df.loc[2].values[1:].astype(np.float32)
        epsh_values = df.loc[3].values[1:].astype(np.float32)
        
        # Extract density values (target)
        density_values = df.loc[4].values[1:].astype(np.float32)
        
        # Stack features: shape (n_samples, 4)
        features = np.column_stack([sigc_values, sigh_values, epsc_values, epsh_values])
        
        return features, density_values
    
    @staticmethod
    def _load_unified_dataset() -> Tuple[np.ndarray, np.ndarray]:
        """Load the unified Dataset.csv file with column-based format.
        
        Expected format (columns: SigC, SigH, EpsC, EpsH, density):
        - Header row with column names
        - Each row is a sample with 5 values
        
        Returns:
            Tuple of (features, targets) arrays of shape (n_samples, 4) and (n_samples,)
        """
        df = pd.read_csv("dataset_combined_cleaned_new.csv")
        
        # Extract features and targets
        features = df[["SigC", "SigH", "EpsC", "EpsH"]].values.astype(np.float32)
        targets = df["density"].values.astype(np.float32)
        
        print(f"[Unified Dataset] Loaded {len(features)} samples from Dataset.csv")
        
        return features, targets
