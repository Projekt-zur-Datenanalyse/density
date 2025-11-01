"""Analyze data ranges and put RMSE results into perspective."""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import ChemicalDensityDataLoader


def analyze_data_ranges(data_dir: str = "."):
    """Analyze feature and target ranges in the dataset.
    
    Args:
        data_dir: Directory containing the CSV files
    """
    data_loader = ChemicalDensityDataLoader(data_dir)
    
    # Auto-discover dataset files
    dataset_paths = sorted(Path(data_dir).glob("Dataset_*.csv"))
    dataset_paths = [str(p) for p in dataset_paths]
    
    if not dataset_paths:
        print("ERROR: No Dataset_*.csv files found!")
        return
    
    # Load all data
    all_features = []
    all_targets = []
    
    for dataset_path in dataset_paths:
        features, targets = data_loader._load_single_dataset(dataset_path)
        all_features.append(features)
        all_targets.append(targets)
    
    # Combine all datasets
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    feature_names = ["SigC", "SigH", "EpsC", "EpsH"]
    
    print("\n" + "=" * 80)
    print("DATA RANGE ANALYSIS - Before Normalization")
    print("=" * 80)
    
    # Feature ranges
    print("\nFEATURE RANGES:")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        min_val = all_features[:, i].min()
        max_val = all_features[:, i].max()
        mean_val = all_features[:, i].mean()
        std_val = all_features[:, i].std()
        range_val = max_val - min_val
        
        print(f"\n{name}:")
        print(f"  Min:   {min_val:.6f}")
        print(f"  Max:   {max_val:.6f}")
        print(f"  Range: {range_val:.6f}")
        print(f"  Mean:  {mean_val:.6f}")
        print(f"  Std:   {std_val:.6f}")
    
    # Target (Density) range
    print("\n" + "-" * 80)
    print("\nTARGET (DENSITY) RANGE:")
    print("-" * 80)
    target_min = all_targets.min()
    target_max = all_targets.max()
    target_range = target_max - target_min
    target_mean = all_targets.mean()
    target_std = all_targets.std()
    
    print(f"\nDensity:")
    print(f"  Min:   {target_min:.6f}")
    print(f"  Max:   {target_max:.6f}")
    print(f"  Range: {target_range:.6f}")
    print(f"  Mean:  {target_mean:.6f}")
    print(f"  Std:   {target_std:.6f}")
    
    # RMSE interpretation
    print("\n" + "=" * 80)
    print("RMSE INTERPRETATION")
    print("=" * 80)
    
    print(f"\nBaseline metrics for density prediction:")
    print(f"  Range of density values: {target_min:.2f} to {target_max:.2f} kg/m³")
    print(f"  Total range: {target_range:.2f} kg/m³")
    print(f"  Mean density: {target_mean:.2f} kg/m³")
    print(f"  Std deviation: {target_std:.2f} kg/m³")
    
    # Naive baseline (predicting mean)
    naive_rmse = target_std
    print(f"\n  Naive baseline (always predict mean): RMSE = {naive_rmse:.2f} kg/m³")
    
    # Example RMSE values and interpretation
    test_rmses = [50, 100, 150, 200, 250]
    print(f"\nExample RMSE interpretations:")
    for rmse in test_rmses:
        pct_range = (rmse / target_range) * 100
        pct_std = (rmse / target_std) * 100
        pct_mean = (rmse / target_mean) * 100
        print(f"  RMSE = {rmse:.0f} kg/m³: {pct_range:.1f}% of range, {pct_std:.1f}% of std, {pct_mean:.1f}% of mean")
    
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nTotal samples: {len(all_features)}")
    print(f"Features shape: {all_features.shape}")
    print(f"Targets shape: {all_targets.shape}")
    
    # Min/Max table for quick reference
    print("\n" + "-" * 80)
    print("QUICK REFERENCE - MIN/MAX VALUES:")
    print("-" * 80)
    print(f"\n{'Feature':<15} {'Min':<15} {'Max':<15} {'Range':<15}")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {all_features[:, i].min():<15.6f} {all_features[:, i].max():<15.6f} {all_features[:, i].max() - all_features[:, i].min():<15.6f}")
    print(f"{'Density':<15} {target_min:<15.2f} {target_max:<15.2f} {target_range:<15.2f}")
    
    return {
        'features': all_features,
        'targets': all_targets,
        'feature_names': feature_names,
        'ranges': {
            name: {
                'min': all_features[:, i].min(),
                'max': all_features[:, i].max(),
                'range': all_features[:, i].max() - all_features[:, i].min(),
                'mean': all_features[:, i].mean(),
                'std': all_features[:, i].std(),
            }
            for i, name in enumerate(feature_names)
        },
        'density': {
            'min': target_min,
            'max': target_max,
            'range': target_range,
            'mean': target_mean,
            'std': target_std,
        }
    }


if __name__ == "__main__":
    data_info = analyze_data_ranges()
