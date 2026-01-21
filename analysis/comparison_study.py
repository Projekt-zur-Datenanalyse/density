"""
Comprehensive comparison study: Big MLP vs Small MLP architectures.

This module implements systematic comparisons of:
1. Big MLP (4→512→2048→254→64→4→1, ~1.6M params) vs Small MLP (4→16→32→8→1, ~900 params)
2. With/without normalization for both architectures
3. Different activation functions (relu, leakyrelu, silu, swiglu)
4. Full dataset vs density-constrained training (600-750 kg/m³)
5. All meaningful combinations for correlation analysis

Key experimental factors:
- Model size: big (1.6M params) vs small (900 params)
- Normalization: with vs without
- Activation: relu, leakyrelu, silu, swiglu
- Data constraint: full vs constrained (600-750 kg/m³)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from itertools import product

# Our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader, Dataset
from core.trainer import Trainer
from core.utils import get_device
from models.mlp import MLP

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class ComparisonConfig:
    """Configuration for comparison experiments."""
    
    # Output directory
    output_dir: str = "results_comparison"
    
    # Dataset
    data_path: str = "dataset.csv"
    validation_split: float = 0.15
    test_split: float = 0.10
    batch_size: int = 64
    seed: int = 46
    
    # Training parameters
    num_epochs: int = 300
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Architectures
    big_hidden_dims: List[int] = field(default_factory=lambda: [512, 2048, 254, 64, 4])
    small_hidden_dims: List[int] = field(default_factory=lambda: [16, 32, 8])
    
    # Activations to test
    activations: List[str] = field(default_factory=lambda: ["relu", "leakyrelu", "silu"])
    
    # Experiment flags
    run_size_comparison: bool = True          # Big vs Small under same conditions
    run_normalization_ablation: bool = True   # With vs Without normalization
    run_activation_comparison: bool = True    # Different activations
    run_data_constraint_study: bool = True    # Full vs Constrained data
    
    # Density constraint for constrained training
    density_min: float = 600.0
    density_max: float = 750.0
    
    # Device
    device: str = "auto"


class BigMLP(nn.Module):
    """Big MLP architecture (~1.6M parameters).
    
    Architecture: 4 → 512 → 2048 → 254 → 64 → 4 → 1
    Configurable activation function.
    """
    
    def __init__(self, activation: str = "leakyrelu"):
        super().__init__()
        
        # Get activation function
        act_fn = self._get_activation(activation)
        
        self.network = nn.Sequential(
            nn.Linear(4, 512),
            act_fn(),
            nn.Linear(512, 2048),
            act_fn(),
            nn.Linear(2048, 254),
            act_fn(),
            nn.Linear(254, 64),
            act_fn(),
            nn.Linear(64, 4),
            act_fn(),
            nn.Linear(4, 1),
        )
        
        self.num_params = sum(p.numel() for p in self.parameters())
        self.activation = activation
    
    def _get_activation(self, name: str):
        """Get activation class by name."""
        activations = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "silu": nn.SiLU,
        }
        return activations.get(name, nn.LeakyReLU)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_test_data_81() -> pd.DataFrame:
    """Load the 81-sample test data (density range ~642-741 kg/m³)."""
    from analysis.raw_test_data import test_data
    return pd.DataFrame(test_data)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive regression metrics."""
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    max_error = np.max(abs_errors)
    mre = np.mean(abs_errors / np.abs(targets)) * 100
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "mre_percent": mre,
        "r2": r2,
    }


class ComparisonStudy:
    """Run comprehensive comparison experiments with meaningful plots."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.device = get_device(config.device)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / f"comparison_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store all results
        self.results: Dict[str, Any] = {
            "config": self._config_to_dict(config),
            "experiments": {},
            "summary": [],
        }
        
        # Cache for loaded data
        self._data_cache = {}
        
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
    
    def _config_to_dict(self, config: ComparisonConfig) -> Dict:
        """Convert config to serializable dict."""
        return {k: v for k, v in config.__dict__.items()}
    
    def _get_data_key(self, normalize: bool, constrained: bool) -> str:
        """Get cache key for data configuration."""
        return f"norm={normalize}_const={constrained}"
    
    def load_data(self, normalize: bool = True, constrained: bool = False) -> Tuple:
        """Load data with caching."""
        key = self._get_data_key(normalize, constrained)
        
        if key in self._data_cache:
            return self._data_cache[key]
        
        loader = DataLoader(self.config.data_path)
        
        if constrained:
            df = pd.read_csv(self.config.data_path)
            mask = (df["density"] >= self.config.density_min) & \
                   (df["density"] <= self.config.density_max)
            df_constrained = df[mask].reset_index(drop=True)
            temp_path = self.output_dir / f"temp_constrained_{key}.csv"
            df_constrained.to_csv(temp_path, index=False)
            loader = DataLoader(str(temp_path))
        
        result = loader.load(
            validation_split=self.config.validation_split,
            test_split=self.config.test_split,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            normalize=normalize,
        )
        
        self._data_cache[key] = result
        return result
    
    def create_model(self, size: str, activation: str) -> nn.Module:
        """Create model by size and activation."""
        if size == "big":
            return BigMLP(activation=activation)
        else:  # small
            return MLP(
                hidden_dims=self.config.small_hidden_dims,
                activation=activation,
            )
    
    def train_model(self, model: nn.Module, train_loader, val_loader, 
                    experiment_name: str, verbose: int = 1) -> Dict[str, Any]:
        """Train a model and return results."""
        
        print(f"\n{'='*60}")
        print(f"Training: {experiment_name}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'='*60}")
        
        trainer = Trainer(
            model=model,
            device=str(self.device),
            checkpoint_dir=str(self.output_dir / experiment_name / "checkpoints"),
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            scheduler_name="cosine",
            verbose=verbose,
        )
        
        return history
    
    def evaluate_model(self, model: nn.Module, test_loader, stats: Dict,
                       normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Evaluate model on test set."""
        model.eval()
        model.to(self.device)
        
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                preds = model(features)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        norm_stats = stats.get("normalization", stats)
        if normalize and norm_stats.get("target_std") is not None:
            predictions = predictions * norm_stats["target_std"] + norm_stats["target_mean"]
            targets = targets * norm_stats["target_std"] + norm_stats["target_mean"]
        
        return predictions, targets, compute_metrics(predictions, targets)
    
    def evaluate_on_test81(self, model: nn.Module, stats: Dict,
                           normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Evaluate model on the 81-sample constrained test set."""
        test_df = get_test_data_81()
        features = test_df[["SigC", "SigH", "EpsC", "EpsH"]].values.astype(np.float32)
        targets = test_df["density"].values.astype(np.float32)
        
        norm_stats = stats.get("normalization", stats)
        
        if normalize and norm_stats.get("feature_mean") is not None:
            feature_mean = np.array(norm_stats["feature_mean"])
            feature_std = np.array(norm_stats["feature_std"])
            features = (features - feature_mean) / feature_std
        
        model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            X = torch.from_numpy(features).float().to(self.device)
            predictions = model(X).cpu().numpy().flatten()
        
        if normalize and norm_stats.get("target_std") is not None:
            predictions = predictions * norm_stats["target_std"] + norm_stats["target_mean"]
        
        return predictions, targets, compute_metrics(predictions, targets)
    
    def run_single_experiment(self, size: str, activation: str, 
                               normalize: bool, constrained: bool) -> Dict:
        """Run a single experiment with given configuration."""
        
        exp_name = f"{size}_{activation}_norm={normalize}_const={constrained}"
        
        # Load data
        train_loader, val_loader, test_loader, stats = self.load_data(
            normalize=normalize, constrained=constrained
        )
        
        # Create and train model
        model = self.create_model(size, activation)
        history = self.train_model(model, train_loader, val_loader, exp_name, verbose=1)
        
        # Evaluate
        preds_int, tgts_int, metrics_int = self.evaluate_model(
            model, test_loader, stats, normalize=normalize
        )
        preds_81, tgts_81, metrics_81 = self.evaluate_on_test81(
            model, stats, normalize=normalize
        )
        
        result = {
            "experiment_name": exp_name,
            "size": size,
            "activation": activation,
            "normalize": normalize,
            "constrained": constrained,
            "num_params": sum(p.numel() for p in model.parameters()),
            "history": history,
            "metrics_internal": metrics_int,
            "metrics_test81": metrics_81,
            "predictions_internal": preds_int.tolist(),
            "targets_internal": tgts_int.tolist(),
            "predictions_test81": preds_81.tolist(),
            "targets_test81": tgts_81.tolist(),
        }
        
        print(f"\n{exp_name} Results:")
        print(f"  Internal - RMSE: {metrics_int['rmse']:.2f}, R²: {metrics_int['r2']:.4f}")
        print(f"  Test-81  - RMSE: {metrics_81['rmse']:.2f}, R²: {metrics_81['r2']:.4f}, MRE: {metrics_81['mre_percent']:.2f}%")
        
        return result
    
    def run_all_experiments(self) -> Dict:
        """Run all comparison experiments."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON STUDY: Big vs Small MLP")
        print("="*80)
        
        all_results = []
        
        # Define experiment combinations
        sizes = ["big", "small"]
        activations = self.config.activations
        normalizations = [True, False]
        constraints = [False, True]  # Full data first, then constrained
        
        # Run selected experiment combinations based on config flags
        experiments_to_run = []
        
        if self.config.run_size_comparison:
            # Big vs Small with same activation, normalization on full data
            for act in activations:
                experiments_to_run.append(("big", act, True, False))
                experiments_to_run.append(("small", act, True, False))
        
        if self.config.run_normalization_ablation:
            # With vs Without normalization for both sizes
            for size in sizes:
                for act in ["leakyrelu"]:  # Use one activation for this comparison
                    experiments_to_run.append((size, act, True, False))
                    experiments_to_run.append((size, act, False, False))
        
        if self.config.run_data_constraint_study:
            # Full vs Constrained data for both sizes
            for size in sizes:
                for act in ["leakyrelu"]:
                    experiments_to_run.append((size, act, True, False))
                    experiments_to_run.append((size, act, True, True))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_experiments = []
        for exp in experiments_to_run:
            if exp not in seen:
                seen.add(exp)
                unique_experiments.append(exp)
        
        print(f"\nRunning {len(unique_experiments)} experiments...")
        
        for size, activation, normalize, constrained in unique_experiments:
            result = self.run_single_experiment(size, activation, normalize, constrained)
            all_results.append(result)
            self.results["experiments"][result["experiment_name"]] = result
        
        self.results["summary"] = all_results
        
        # Save results
        self._save_results()
        
        # Generate comprehensive plots
        self._generate_all_plots()
        
        # Print summary table
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON and CSV."""
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for result in self.results["summary"]:
            summary_data.append({
                "experiment": result["experiment_name"],
                "size": result["size"],
                "activation": result["activation"],
                "normalize": result["normalize"],
                "constrained": result["constrained"],
                "num_params": result["num_params"],
                "internal_rmse": result["metrics_internal"]["rmse"],
                "internal_r2": result["metrics_internal"]["r2"],
                "internal_mae": result["metrics_internal"]["mae"],
                "test81_rmse": result["metrics_test81"]["rmse"],
                "test81_r2": result["metrics_test81"]["r2"],
                "test81_mre": result["metrics_test81"]["mre_percent"],
                "test81_mae": result["metrics_test81"]["mae"],
            })
        
        pd.DataFrame(summary_data).to_csv(self.output_dir / "summary.csv", index=False)
        print(f"\nResults saved to: {results_path}")
    
    def _print_summary(self):
        """Print summary table."""
        print("\n" + "="*120)
        print("SUMMARY TABLE")
        print("="*120)
        print(f"{'Experiment':<45} {'Size':<6} {'Act':<10} {'Norm':<6} {'Const':<6} {'Params':>10} {'Int RMSE':>10} {'T81 RMSE':>10} {'T81 R²':>8}")
        print("-"*120)
        
        for r in sorted(self.results["summary"], key=lambda x: x["metrics_test81"]["rmse"]):
            print(f"{r['experiment_name']:<45} {r['size']:<6} {r['activation']:<10} "
                  f"{str(r['normalize']):<6} {str(r['constrained']):<6} {r['num_params']:>10,} "
                  f"{r['metrics_internal']['rmse']:>10.2f} {r['metrics_test81']['rmse']:>10.2f} "
                  f"{r['metrics_test81']['r2']:>8.4f}")
    
    def _generate_all_plots(self):
        """Generate all comparison plots."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame([{
            "experiment": r["experiment_name"],
            "size": r["size"],
            "activation": r["activation"],
            "normalize": r["normalize"],
            "constrained": r["constrained"],
            "num_params": r["num_params"],
            "internal_rmse": r["metrics_internal"]["rmse"],
            "internal_r2": r["metrics_internal"]["r2"],
            "test81_rmse": r["metrics_test81"]["rmse"],
            "test81_r2": r["metrics_test81"]["r2"],
            "test81_mre": r["metrics_test81"]["mre_percent"],
        } for r in self.results["summary"]])
        
        # 1. Size comparison under same conditions
        self._plot_size_comparison(df, plots_dir)
        
        # 2. Normalization impact
        self._plot_normalization_impact(df, plots_dir)
        
        # 3. Data constraint impact
        self._plot_constraint_impact(df, plots_dir)
        
        # 4. Activation function comparison
        self._plot_activation_comparison(df, plots_dir)
        
        # 5. Parameter efficiency
        self._plot_parameter_efficiency(df, plots_dir)
        
        # 6. Training curves by category
        self._plot_training_curves(plots_dir)
        
        # 7. Prediction scatter plots
        self._plot_prediction_scatter(plots_dir)
        
        # 8. Error distribution comparison
        self._plot_error_distributions(plots_dir)
        
        # 9. Correlation heatmap
        self._plot_correlation_heatmap(df, plots_dir)
        
        # 10. Multi-factor analysis
        self._plot_multi_factor_analysis(df, plots_dir)
        
        print(f"Plots saved to: {plots_dir}")
    
    def _plot_size_comparison(self, df: pd.DataFrame, plots_dir: Path):
        """Compare Big vs Small model performance under same conditions."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Filter for normalized, unconstrained experiments
        mask = (df["normalize"] == True) & (df["constrained"] == False)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            plt.close()
            return
        
        # Group by activation and size
        for ax, metric, title in zip(axes, 
            ["test81_rmse", "test81_r2", "internal_rmse"],
            ["Test-81 RMSE (kg/m³)", "Test-81 R²", "Internal Test RMSE (kg/m³)"]):
            
            pivot = subset.pivot_table(index="activation", columns="size", values=metric, aggfunc="first")
            if len(pivot) > 0:
                pivot.plot(kind="bar", ax=ax, rot=0, width=0.7)
                ax.set_xlabel("Activation Function")
                ax.set_ylabel(title)
                ax.set_title(f"Big vs Small: {title}")
                ax.legend(title="Model Size")
                
                # Add value labels
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "01_size_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_normalization_impact(self, df: pd.DataFrame, plots_dir: Path):
        """Compare performance with vs without normalization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Filter for unconstrained experiments with leakyrelu
        mask = (df["constrained"] == False) & (df["activation"] == "leakyrelu")
        subset = df[mask].copy()
        
        if len(subset) < 2:
            plt.close()
            return
        
        # Plot for each model size
        for ax, metric, title in zip(axes,
            ["test81_rmse", "internal_rmse"],
            ["Test-81 RMSE (kg/m³)", "Internal RMSE (kg/m³)"]):
            
            pivot = subset.pivot_table(index="size", columns="normalize", values=metric, aggfunc="first")
            if len(pivot) > 0:
                pivot.columns = ["Without Norm", "With Norm"]
                pivot.plot(kind="bar", ax=ax, rot=0, color=["#e74c3c", "#2ecc71"], width=0.6)
                ax.set_xlabel("Model Size")
                ax.set_ylabel(title)
                ax.set_title(f"Normalization Impact: {title}")
                ax.legend(title="Normalization")
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "02_normalization_impact.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_constraint_impact(self, df: pd.DataFrame, plots_dir: Path):
        """Compare performance on full vs constrained training data."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Filter for normalized experiments with leakyrelu
        mask = (df["normalize"] == True) & (df["activation"] == "leakyrelu")
        subset = df[mask].copy()
        
        if len(subset) < 2:
            plt.close()
            return
        
        for ax, metric, title in zip(axes,
            ["test81_rmse", "internal_rmse"],
            ["Test-81 RMSE (kg/m³)\n(Constrained Test Set)", "Internal RMSE (kg/m³)\n(Matched Distribution)"]):
            
            pivot = subset.pivot_table(index="size", columns="constrained", values=metric, aggfunc="first")
            if len(pivot) > 0:
                pivot.columns = ["Full Data", "Constrained (600-750)"]
                pivot.plot(kind="bar", ax=ax, rot=0, color=["#3498db", "#9b59b6"], width=0.6)
                ax.set_xlabel("Model Size")
                ax.set_ylabel(title)
                ax.set_title(f"Training Data Impact: {title}")
                ax.legend(title="Training Data")
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "03_data_constraint_impact.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_activation_comparison(self, df: pd.DataFrame, plots_dir: Path):
        """Compare different activation functions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Filter for normalized, unconstrained experiments
        mask = (df["normalize"] == True) & (df["constrained"] == False)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            plt.close()
            return
        
        metrics = [
            ("test81_rmse", "Test-81 RMSE (kg/m³)"),
            ("test81_r2", "Test-81 R²"),
            ("internal_rmse", "Internal RMSE (kg/m³)"),
            ("internal_r2", "Internal R²"),
        ]
        
        for ax, (metric, title) in zip(axes.flat, metrics):
            for size in ["big", "small"]:
                size_data = subset[subset["size"] == size]
                if len(size_data) > 0:
                    ax.plot(size_data["activation"], size_data[metric], 
                           'o-', label=size.capitalize(), markersize=10, linewidth=2)
            
            ax.set_xlabel("Activation Function")
            ax.set_ylabel(title)
            ax.set_title(f"Activation Comparison: {title}")
            ax.legend(title="Model Size")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "04_activation_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_efficiency(self, df: pd.DataFrame, plots_dir: Path):
        """Plot RMSE vs number of parameters with categorical markers."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create marker and color mappings
        markers = {"big": "s", "small": "o"}
        norm_colors = {True: "#2ecc71", False: "#e74c3c"}
        
        for _, row in df.iterrows():
            ax.scatter(row["num_params"], row["test81_rmse"],
                      s=200 if row["constrained"] else 100,
                      marker=markers[row["size"]],
                      c=norm_colors[row["normalize"]],
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=1)
            
            # Add label
            offset = (0.1, 0.3) if row["size"] == "big" else (-0.1, -0.5)
            ax.annotate(f"{row['activation'][:4]}", 
                       (row["num_params"], row["test81_rmse"]),
                       fontsize=7, ha='center',
                       xytext=(5, 5), textcoords='offset points')
        
        # Add legend elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Big Model'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Small Model'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='With Norm'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Without Norm'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xscale('log')
        ax.set_xlabel("Number of Parameters (log scale)")
        ax.set_ylabel("Test-81 RMSE (kg/m³)")
        ax.set_title("Parameter Efficiency: RMSE vs Model Size")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "05_parameter_efficiency.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self, plots_dir: Path):
        """Plot training curves grouped by comparison type."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10.colors
        
        # Group experiments by type
        groups = {
            "Big vs Small (normalized, full data)": [],
            "Normalization Impact": [],
            "Data Constraint Impact": [],
            "Activation Comparison": [],
        }
        
        for exp_name, exp_data in self.results["experiments"].items():
            if "norm=True" in exp_name and "const=False" in exp_name:
                groups["Big vs Small (normalized, full data)"].append((exp_name, exp_data))
            if "leakyrelu" in exp_name and "const=False" in exp_name:
                groups["Normalization Impact"].append((exp_name, exp_data))
            if "leakyrelu" in exp_name and "norm=True" in exp_name:
                groups["Data Constraint Impact"].append((exp_name, exp_data))
            if "norm=True" in exp_name and "const=False" in exp_name:
                groups["Activation Comparison"].append((exp_name, exp_data))
        
        for ax, (group_name, experiments) in zip(axes.flat, groups.items()):
            for i, (exp_name, exp_data) in enumerate(experiments[:6]):  # Limit to 6
                history = exp_data.get("history", {})
                if "val_loss" in history:
                    epochs = range(1, len(history["val_loss"]) + 1)
                    label = exp_name.replace("_norm=True", "").replace("_norm=False", "").replace("_const=False", "").replace("_const=True", " (const)")
                    ax.plot(epochs, history["val_loss"], 
                           label=label[:25], 
                           color=colors[i % len(colors)],
                           linewidth=1.5)
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_title(group_name)
            ax.legend(fontsize=7, loc='upper right')
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "06_training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self, plots_dir: Path):
        """Plot predictions vs actual for best models in each category."""
        # Select diverse experiments
        experiments_to_plot = []
        seen_configs = set()
        
        for exp_name, exp_data in self.results["experiments"].items():
            key = (exp_data["size"], exp_data["normalize"])
            if key not in seen_configs:
                seen_configs.add(key)
                experiments_to_plot.append((exp_name, exp_data))
        
        n_plots = min(len(experiments_to_plot), 6)
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for ax, (exp_name, exp_data) in zip(axes.flat[:n_plots], experiments_to_plot[:n_plots]):
            preds = np.array(exp_data["predictions_test81"])
            targets = np.array(exp_data["targets_test81"])
            
            ax.scatter(targets, preds, alpha=0.6, s=40, c='#3498db')
            
            lims = [min(targets.min(), preds.min()) - 5, max(targets.max(), preds.max()) + 5]
            ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
            
            ax.set_xlabel("Actual Density (kg/m³)")
            ax.set_ylabel("Predicted Density (kg/m³)")
            title = f"{exp_data['size'].upper()} | Norm={exp_data['normalize']}"
            ax.set_title(f"{title}\nRMSE: {exp_data['metrics_test81']['rmse']:.2f} kg/m³")
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        for ax in axes.flat[n_plots:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "07_prediction_scatter.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distributions(self, plots_dir: Path):
        """Plot error distributions comparing key configurations."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Group by size
        for ax, size in zip(axes, ["big", "small"]):
            errors_data = []
            labels = []
            
            for exp_name, exp_data in self.results["experiments"].items():
                if exp_data["size"] == size:
                    preds = np.array(exp_data["predictions_test81"])
                    targets = np.array(exp_data["targets_test81"])
                    errors = preds - targets
                    errors_data.append(errors)
                    
                    # Create short label
                    label = f"{exp_data['activation'][:4]}"
                    if not exp_data["normalize"]:
                        label += " (no norm)"
                    if exp_data["constrained"]:
                        label += " (const)"
                    labels.append(label)
            
            if len(errors_data) > 0:
                bp = ax.boxplot(errors_data, labels=labels, vert=True, patch_artist=True)
                
                colors = plt.cm.Set2.colors
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
                ax.set_ylabel("Prediction Error (kg/m³)")
                ax.set_title(f"Error Distribution: {size.upper()} Model")
                ax.tick_params(axis='x', rotation=30)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "08_error_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, plots_dir: Path):
        """Plot correlation between experimental factors and performance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for correlation
        df_corr = df.copy()
        df_corr["size_numeric"] = df_corr["size"].map({"small": 0, "big": 1})
        df_corr["normalize_numeric"] = df_corr["normalize"].astype(int)
        df_corr["constrained_numeric"] = df_corr["constrained"].astype(int)
        df_corr["log_params"] = np.log10(df_corr["num_params"])
        
        # Encode activation
        act_map = {"relu": 0, "leakyrelu": 1, "silu": 2, "swiglu": 3}
        df_corr["activation_numeric"] = df_corr["activation"].map(act_map)
        
        # Select columns for correlation
        corr_cols = ["size_numeric", "normalize_numeric", "constrained_numeric", 
                     "log_params", "test81_rmse", "test81_r2", "internal_rmse", "internal_r2"]
        
        # Filter valid columns
        valid_cols = [c for c in corr_cols if c in df_corr.columns]
        
        if len(valid_cols) > 2:
            corr_matrix = df_corr[valid_cols].corr()
            
            # Rename for readability
            rename_map = {
                "size_numeric": "Model Size",
                "normalize_numeric": "Normalization",
                "constrained_numeric": "Constrained Data",
                "log_params": "Log(Parameters)",
                "test81_rmse": "Test-81 RMSE",
                "test81_r2": "Test-81 R²",
                "internal_rmse": "Internal RMSE",
                "internal_r2": "Internal R²",
            }
            corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
            
            sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, 
                       fmt=".2f", ax=ax, square=True, linewidths=0.5)
            ax.set_title("Correlation Between Factors and Performance Metrics")
        
        plt.tight_layout()
        plt.savefig(plots_dir / "09_correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_multi_factor_analysis(self, df: pd.DataFrame, plots_dir: Path):
        """Multi-factor analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Size vs Normalization interaction
        ax = axes[0, 0]
        for norm in [True, False]:
            subset = df[df["normalize"] == norm]
            for size in ["big", "small"]:
                size_data = subset[subset["size"] == size]
                if len(size_data) > 0:
                    label = f"{size.capitalize()}, {'Norm' if norm else 'No Norm'}"
                    marker = 'o' if size == "small" else 's'
                    color = '#2ecc71' if norm else '#e74c3c'
                    ax.scatter(size_data["num_params"], size_data["test81_rmse"],
                              s=150, marker=marker, c=color, label=label, alpha=0.7, edgecolors='black')
        
        ax.set_xscale('log')
        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("Test-81 RMSE (kg/m³)")
        ax.set_title("Size × Normalization Interaction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Best configuration per category
        ax = axes[0, 1]
        categories = []
        values = []
        colors = []
        
        for size in ["big", "small"]:
            for norm in [True, False]:
                subset = df[(df["size"] == size) & (df["normalize"] == norm) & (df["constrained"] == False)]
                if len(subset) > 0:
                    best = subset.loc[subset["test81_rmse"].idxmin()]
                    categories.append(f"{size[:1].upper()}, {'N' if norm else 'NN'}")
                    values.append(best["test81_rmse"])
                    colors.append('#2ecc71' if norm else '#e74c3c')
        
        if len(categories) > 0:
            bars = ax.bar(categories, values, color=colors, edgecolor='black')
            ax.set_ylabel("Best Test-81 RMSE (kg/m³)")
            ax.set_title("Best Performance by Configuration")
            ax.bar_label(bars, fmt='%.2f')
        
        # 3. Improvement from normalization
        ax = axes[1, 0]
        improvements = []
        labels = []
        
        for size in ["big", "small"]:
            norm_data = df[(df["size"] == size) & (df["normalize"] == True) & (df["constrained"] == False)]
            no_norm_data = df[(df["size"] == size) & (df["normalize"] == False) & (df["constrained"] == False)]
            
            if len(norm_data) > 0 and len(no_norm_data) > 0:
                norm_best = norm_data["test81_rmse"].min()
                no_norm_best = no_norm_data["test81_rmse"].min()
                improvement = ((no_norm_best - norm_best) / no_norm_best) * 100
                improvements.append(improvement)
                labels.append(size.capitalize())
        
        if len(improvements) > 0:
            bars = ax.bar(labels, improvements, color=['#3498db', '#9b59b6'], edgecolor='black')
            ax.set_ylabel("RMSE Improvement (%)")
            ax.set_title("Performance Improvement from Normalization")
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.bar_label(bars, fmt='%.1f%%')
        
        # 4. Ranking summary
        ax = axes[1, 1]
        df_sorted = df.sort_values("test81_rmse").head(8)
        
        y_pos = range(len(df_sorted))
        colors = ['#2ecc71' if r["normalize"] else '#e74c3c' for _, r in df_sorted.iterrows()]
        
        bars = ax.barh(y_pos, df_sorted["test81_rmse"], color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['size'][:1].upper()}-{r['activation'][:4]}" 
                           for _, r in df_sorted.iterrows()], fontsize=9)
        ax.set_xlabel("Test-81 RMSE (kg/m³)")
        ax.set_title("Top 8 Configurations (Green=Norm, Red=No Norm)")
        ax.invert_yaxis()
        
        for bar, val in zip(bars, df_sorted["test81_rmse"]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "10_multi_factor_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()


def run_comparison_study(config: Optional[ComparisonConfig] = None) -> Dict:
    """Main entry point for running comparison study."""
    if config is None:
        config = ComparisonConfig()
    
    study = ComparisonStudy(config)
    results = study.run_all_experiments()
    
    return results


if __name__ == "__main__":
    config = ComparisonConfig(num_epochs=100)
    results = run_comparison_study(config)
