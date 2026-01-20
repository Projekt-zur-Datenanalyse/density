"""
Plotting utilities for visualization.

This module provides functions for creating visualizations
of data, training progress, and model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class Plotter:
    """Utility class for creating plots."""
    
    def __init__(
        self,
        output_dir: str = "plots",
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
    ):
        """Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        
        try:
            plt.style.use(style)
        except Exception:
            pass
    
    def training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        filename: str = "training_history.png",
    ) -> str:
        """Plot training history.
        
        Args:
            history: Dictionary with train_loss and val_loss lists
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history.get("train_loss", [])) + 1)
        
        if "train_loss" in history:
            ax.plot(epochs, history["train_loss"], label="Training Loss", alpha=0.8)
        
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], label="Validation Loss", alpha=0.8)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        
        return str(output_path)
    
    def predictions_vs_actual(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Predictions vs Actual",
        filename: str = "predictions_vs_actual.png",
        unit: str = "kg/m³",
    ) -> str:
        """Plot predictions vs actual values.
        
        Args:
            predictions: Predicted values
            targets: Actual values
            title: Plot title
            filename: Output filename
            unit: Unit label
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.scatter(targets, predictions, alpha=0.5, s=20)
        
        # Diagonal line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect")
        
        ax.set_xlabel(f"Actual ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        
        return str(output_path)
    
    def residual_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Residual Plot",
        filename: str = "residuals.png",
        unit: str = "kg/m³",
    ) -> str:
        """Create residual plot.
        
        Args:
            predictions: Predicted values
            targets: Actual values
            title: Plot title
            filename: Output filename
            unit: Unit label
            
        Returns:
            Path to saved figure
        """
        residuals = predictions - targets
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        # Residuals vs predicted
        axes[0].scatter(predictions, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel(f"Predicted ({unit})")
        axes[0].set_ylabel(f"Residual ({unit})")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel(f"Residual ({unit})")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residual Distribution")
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        
        return str(output_path)
    
    def uncertainty_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
        title: str = "Predictions with Uncertainty",
        filename: str = "uncertainty.png",
        unit: str = "kg/m³",
    ) -> str:
        """Plot predictions with uncertainty estimates.
        
        Args:
            predictions: Mean predictions
            targets: Actual values
            uncertainties: Standard deviation of predictions
            title: Plot title
            filename: Output filename
            unit: Unit label
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by uncertainty for better visualization
        sort_idx = np.argsort(uncertainties)
        
        scatter = ax.scatter(
            targets[sort_idx],
            predictions[sort_idx],
            c=uncertainties[sort_idx],
            cmap="viridis",
            alpha=0.6,
            s=30,
        )
        
        # Diagonal line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect")
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"Uncertainty ({unit})")
        
        ax.set_xlabel(f"Actual ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        
        return str(output_path)
    
    def model_comparison(
        self,
        results: Dict[str, float],
        metric_name: str = "RMSE (kg/m³)",
        title: str = "Model Comparison",
        filename: str = "model_comparison.png",
    ) -> str:
        """Create model comparison bar chart.
        
        Args:
            results: Dictionary mapping model names to metric values
            metric_name: Name of the metric
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = list(results.keys())
        values = list(results.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.bar(models, values, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
            )
        
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        
        return str(output_path)


def create_comparison_plot(
    results: Dict[str, Dict[str, float]],
    metric: str = "rmse_denormalized",
    output_path: str = "model_comparison.png",
) -> str:
    """Create a quick model comparison plot.
    
    Args:
        results: Dictionary mapping model names to result dictionaries
        metric: Metric to compare
        output_path: Output file path
        
    Returns:
        Path to saved figure
    """
    plotter = Plotter(output_dir=str(Path(output_path).parent))
    
    comparison_data = {
        name: result.get(metric, 0) for name, result in results.items()
    }
    
    return plotter.model_comparison(
        comparison_data,
        filename=Path(output_path).name,
    )


def create_residual_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str = "residuals.png",
    title: str = "Residual Analysis",
) -> str:
    """Create a quick residual plot.
    
    Args:
        predictions: Predicted values
        targets: Actual values
        output_path: Output file path
        title: Plot title
        
    Returns:
        Path to saved figure
    """
    plotter = Plotter(output_dir=str(Path(output_path).parent))
    
    return plotter.residual_plot(
        predictions=predictions,
        targets=targets,
        title=title,
        filename=Path(output_path).name,
    )


__all__ = ["Plotter", "create_comparison_plot", "create_residual_plot"]
