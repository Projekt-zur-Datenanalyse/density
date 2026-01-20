"""
Model analysis and comparison utilities.

This module provides tools for analyzing trained models,
comparing architectures, and evaluating ensemble performance.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class ModelAnalyzer:
    """Analyzer for trained model results."""
    
    def __init__(self, results_dir: str):
        """Initialize the analyzer.
        
        Args:
            results_dir: Directory containing model results
        """
        self.results_dir = Path(results_dir)
    
    def load_test_results(self) -> Dict[str, Any]:
        """Load test results from a results directory.
        
        Returns:
            Dictionary with test results
        """
        results_file = self.results_dir / "test_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")
        
        with open(results_file) as f:
            return json.load(f)
    
    def load_training_history(self) -> Dict[str, Any]:
        """Load training history.
        
        Returns:
            Dictionary with training history
        """
        history_file = self.results_dir / "training_history.json"
        
        if not history_file.exists():
            return {}
        
        with open(history_file) as f:
            return json.load(f)
    
    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration.
        
        Returns:
            Dictionary with model config
        """
        config_file = self.results_dir / "model_config.json"
        
        if not config_file.exists():
            return {}
        
        with open(config_file) as f:
            return json.load(f)
    
    def load_predictions(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load saved predictions.
        
        Returns:
            Dictionary with predictions tensor, or None
        """
        pred_file = self.results_dir / "predictions.pt"
        
        if not pred_file.exists():
            return None
        
        return torch.load(pred_file, map_location='cpu', weights_only=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of model results.
        
        Returns:
            Dictionary with model summary
        """
        return {
            "test_results": self.load_test_results(),
            "model_config": self.load_model_config(),
            "training_history_length": len(
                self.load_training_history().get("train_loss", [])
            ),
        }
    
    def compute_residuals(self) -> Optional[Dict[str, np.ndarray]]:
        """Compute prediction residuals.
        
        Returns:
            Dictionary with residual statistics, or None
        """
        predictions = self.load_predictions()
        
        if predictions is None:
            return None
        
        if "predictions" not in predictions or "targets" not in predictions:
            return None
        
        preds = predictions["predictions"].numpy()
        targets = predictions["targets"].numpy()
        residuals = preds - targets
        
        return {
            "residuals": residuals,
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        }


def compare_models(
    results_dirs: List[str],
    metric: str = "rmse_denormalized",
) -> Dict[str, Any]:
    """Compare multiple trained models.
    
    Args:
        results_dirs: List of result directory paths
        metric: Metric to compare
        
    Returns:
        Comparison results
    """
    results = []
    
    for dir_path in results_dirs:
        analyzer = ModelAnalyzer(dir_path)
        try:
            test_results = analyzer.load_test_results()
            config = analyzer.load_model_config()
            
            results.append({
                "directory": dir_path,
                "architecture": config.get("architecture", "unknown"),
                "metric_value": test_results.get(metric, float("inf")),
                "all_metrics": test_results,
            })
        except Exception as e:
            results.append({
                "directory": dir_path,
                "error": str(e),
            })
    
    # Sort by metric value
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["metric_value"])
    
    return {
        "metric": metric,
        "rankings": valid_results,
        "errors": [r for r in results if "error" in r],
        "best": valid_results[0] if valid_results else None,
    }


def analyze_ensemble_results(ensemble_dir: str) -> Dict[str, Any]:
    """Analyze deep ensemble results.
    
    Args:
        ensemble_dir: Directory containing ensemble results
        
    Returns:
        Ensemble analysis results
    """
    ensemble_dir = Path(ensemble_dir)
    
    # Load ensemble results
    results_file = ensemble_dir / "ensemble_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Ensemble results not found: {results_file}")
    
    with open(results_file) as f:
        ensemble_results = json.load(f)
    
    # Load individual model results
    member_results = []
    member_dirs = sorted(ensemble_dir.glob("model_*"))
    
    for member_dir in member_dirs:
        try:
            analyzer = ModelAnalyzer(member_dir)
            member_results.append({
                "model": member_dir.name,
                "results": analyzer.load_test_results(),
            })
        except Exception:
            pass
    
    # Compute statistics
    if member_results:
        rmse_values = [
            m["results"].get("rmse_denormalized", float("inf"))
            for m in member_results
        ]
        
        individual_stats = {
            "mean_rmse": float(np.mean(rmse_values)),
            "std_rmse": float(np.std(rmse_values)),
            "min_rmse": float(np.min(rmse_values)),
            "max_rmse": float(np.max(rmse_values)),
        }
    else:
        individual_stats = {}
    
    return {
        "ensemble_metrics": ensemble_results,
        "individual_stats": individual_stats,
        "n_members": len(member_results),
        "member_results": member_results,
    }


__all__ = ["ModelAnalyzer", "compare_models", "analyze_ensemble_results"]
