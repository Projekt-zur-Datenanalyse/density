"""
Data analysis and exploration utilities.

This module provides tools for analyzing the chemical density dataset,
including feature distributions, correlations, and statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


class DataAnalyzer:
    """Analyzer for chemical density dataset."""
    
    def __init__(self, data_path: str = "dataset.csv"):
        """Initialize the analyzer.
        
        Args:
            data_path: Path to the dataset
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols = ['SigC', 'SigH', 'EpsC', 'EpsH']
        self.target_col = 'density'
    
    def load_data(self) -> pd.DataFrame:
        """Load and return the dataset."""
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.df is None:
            self.load_data()
        
        return {
            "n_samples": len(self.df),
            "n_features": len(self.feature_cols),
            "features": self.feature_cols,
            "target": self.target_col,
            "missing_values": self.df.isnull().sum().to_dict(),
            "feature_stats": self.df[self.feature_cols].describe().to_dict(),
            "target_stats": self.df[self.target_col].describe().to_dict(),
        }
    
    def get_feature_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get min/max ranges for each feature.
        
        Returns:
            Dictionary mapping feature names to ranges
        """
        if self.df is None:
            self.load_data()
        
        ranges = {}
        for col in self.feature_cols + [self.target_col]:
            ranges[col] = {
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "range": float(self.df[col].max() - self.df[col].min()),
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
            }
        
        return ranges
    
    def get_correlations(self) -> pd.DataFrame:
        """Get correlation matrix.
        
        Returns:
            Correlation matrix DataFrame
        """
        if self.df is None:
            self.load_data()
        
        all_cols = self.feature_cols + [self.target_col]
        return self.df[all_cols].corr()
    
    def get_target_correlations(self) -> Dict[str, float]:
        """Get correlations between features and target.
        
        Returns:
            Dictionary mapping feature names to correlations
        """
        corr_matrix = self.get_correlations()
        
        return {
            col: float(corr_matrix.loc[col, self.target_col])
            for col in self.feature_cols
        }
    
    def get_percentiles(
        self,
        column: str,
        percentiles: List[float] = [0, 5, 10, 25, 50, 75, 90, 95, 100],
    ) -> Dict[int, float]:
        """Get percentile values for a column.
        
        Args:
            column: Column name
            percentiles: List of percentiles to compute
            
        Returns:
            Dictionary mapping percentile to value
        """
        if self.df is None:
            self.load_data()
        
        values = self.df[column].quantile([p / 100 for p in percentiles])
        return {
            int(p): float(v) for p, v in zip(percentiles, values)
        }
    
    def detect_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect outliers in a column.
        
        Args:
            column: Column name
            method: Detection method ("iqr" or "zscore")
            threshold: Threshold for detection
            
        Returns:
            Tuple of (outlier mask, statistics dict)
        """
        if self.df is None:
            self.load_data()
        
        values = self.df[column].values
        
        if method == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (values < lower) | (values > upper)
        elif method == "zscore":
            z = (values - np.mean(values)) / np.std(values)
            outliers = np.abs(z) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        stats = {
            "n_outliers": int(np.sum(outliers)),
            "pct_outliers": float(np.mean(outliers) * 100),
            "method": method,
            "threshold": threshold,
        }
        
        return outliers, stats
    
    def save_report(self, output_path: str = "analysis/data_report.txt"):
        """Save a comprehensive data report.
        
        Args:
            output_path: Path for the report file
        """
        if self.df is None:
            self.load_data()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CHEMICAL DENSITY DATASET ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic info
            summary = self.get_summary()
            f.write(f"Data Path: {self.data_path}\n")
            f.write(f"Samples: {summary['n_samples']}\n")
            f.write(f"Features: {summary['features']}\n")
            f.write(f"Target: {summary['target']}\n\n")
            
            # Feature ranges
            f.write("-" * 60 + "\n")
            f.write("FEATURE RANGES\n")
            f.write("-" * 60 + "\n\n")
            
            ranges = self.get_feature_ranges()
            for col, stats in ranges.items():
                f.write(f"{col}:\n")
                f.write(f"  Min:   {stats['min']:.6f}\n")
                f.write(f"  Max:   {stats['max']:.6f}\n")
                f.write(f"  Range: {stats['range']:.6f}\n")
                f.write(f"  Mean:  {stats['mean']:.6f}\n")
                f.write(f"  Std:   {stats['std']:.6f}\n\n")
            
            # Correlations
            f.write("-" * 60 + "\n")
            f.write("TARGET CORRELATIONS\n")
            f.write("-" * 60 + "\n\n")
            
            corrs = self.get_target_correlations()
            for feat, corr in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
                f.write(f"{feat}: {corr:+.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        print(f"Report saved to: {output_path}")
        return output_path


def analyze_data_ranges(data_path: str = "dataset.csv") -> Dict[str, Any]:
    """Quick function to analyze data ranges.
    
    Args:
        data_path: Path to dataset
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = DataAnalyzer(data_path)
    analyzer.load_data()
    
    return {
        "summary": analyzer.get_summary(),
        "ranges": analyzer.get_feature_ranges(),
        "correlations": analyzer.get_target_correlations(),
    }


__all__ = ["DataAnalyzer", "analyze_data_ranges"]
