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
    
    # Load data using the unified dataset loader (supports new format)
    all_features, all_targets = data_loader._load_unified_dataset()
    
    feature_names = ["SigC", "SigH", "EpsC", "EpsH"]
    
    print("\n" + "=" * 80)
    print("DATA RANGE ANALYSIS - Before Normalization")
    print("=" * 80)
    
    # Feature ranges
    print("\nFEATURE RANGES & STATISTICS:")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        min_val = all_features[:, i].min()
        max_val = all_features[:, i].max()
        mean_val = all_features[:, i].mean()
        std_val = all_features[:, i].std()
        median_val = np.median(all_features[:, i])
        range_val = max_val - min_val
        q1 = np.percentile(all_features[:, i], 25)
        q3 = np.percentile(all_features[:, i], 75)
        iqr = q3 - q1
        cv = (std_val / mean_val) * 100  # Coefficient of variation
        
        print(f"\n{name}:")
        print(f"  Min:        {min_val:.6f}")
        print(f"  Q1 (25%):   {q1:.6f}")
        print(f"  Median:     {median_val:.6f}")
        print(f"  Mean:       {mean_val:.6f}")
        print(f"  Q3 (75%):   {q3:.6f}")
        print(f"  Max:        {max_val:.6f}")
        print(f"  ---")
        print(f"  Range:      {range_val:.6f}")
        print(f"  IQR:        {iqr:.6f}")
        print(f"  Std Dev:    {std_val:.6f}")
        print(f"  Variance:   {std_val**2:.6f}")
        print(f"  Coeff. Var: {cv:.2f}%")
        print(f"  Skewness:   {np.mean((all_features[:, i] - mean_val)**3) / (std_val**3):.6f}")
    
    # Target (Density) range
    print("\n" + "-" * 80)
    print("\nTARGET (DENSITY) RANGE & STATISTICS:")
    print("-" * 80)
    target_min = all_targets.min()
    target_max = all_targets.max()
    target_range = target_max - target_min
    target_mean = all_targets.mean()
    target_std = all_targets.std()
    target_median = np.median(all_targets)
    target_q1 = np.percentile(all_targets, 25)
    target_q3 = np.percentile(all_targets, 75)
    target_iqr = target_q3 - target_q1
    target_cv = (target_std / target_mean) * 100
    target_skew = np.mean((all_targets - target_mean)**3) / (target_std**3)
    target_var = target_std ** 2
    
    print(f"\nDensity:")
    print(f"  Min:        {target_min:.2f}")
    print(f"  Q1 (25%):   {target_q1:.2f}")
    print(f"  Median:     {target_median:.2f}")
    print(f"  Mean:       {target_mean:.2f}")
    print(f"  Q3 (75%):   {target_q3:.2f}")
    print(f"  Max:        {target_max:.2f}")
    print(f"  ---")
    print(f"  Range:      {target_range:.2f}")
    print(f"  IQR:        {target_iqr:.2f}")
    print(f"  Std Dev:    {target_std:.2f}")
    print(f"  Variance:   {target_var:.2f}")
    print(f"  Coeff. Var: {target_cv:.2f}%")
    print(f"  Skewness:   {target_skew:.6f}")
    
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
    print("QUICK REFERENCE - SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"\n{'Feature':<12} {'Min':<12} {'Mean':<12} {'Std':<12} {'Max':<12} {'Coeff.Var':<12}")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        feat_min = all_features[:, i].min()
        feat_mean = all_features[:, i].mean()
        feat_std = all_features[:, i].std()
        feat_max = all_features[:, i].max()
        feat_cv = (feat_std / feat_mean) * 100
        print(f"{name:<12} {feat_min:<12.6f} {feat_mean:<12.6f} {feat_std:<12.6f} {feat_max:<12.6f} {feat_cv:<12.2f}%")
    print(f"{'Density':<12} {target_min:<12.2f} {target_mean:<12.2f} {target_std:<12.2f} {target_max:<12.2f} {target_cv:<12.2f}%")
    
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
