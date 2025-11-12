"""Analysis and visualization of Optuna hyperparameter tuning results.

This script provides tools to:
- Load and analyze tuning results
- Compare different trials
- Generate performance reports
- Extract and export best configurations
- Visualize hyperparameter importance
"""

import json
import argparse
import optuna
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class OptunaResultsAnalyzer:
    """Analyzer for Optuna hyperparameter tuning results."""
    
    def __init__(self, study_dir: str, verbose: bool = True):
        """Initialize the analyzer.
        
        Args:
            study_dir: Directory containing Optuna study database
            verbose: Whether to print detailed output
        """
        self.study_dir = Path(study_dir)
        self.verbose = verbose
        self.summary_file = self.study_dir / "optuna_summary.json"
        self.db_file = self.study_dir / "study.db"
        self.summary = None
        self.study = None
        self.target_std = None
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPTUNA RESULTS ANALYZER")
            print(f"{'='*80}")
            print(f"Results Directory: {self.study_dir}")
        
        # Load summary
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.summary = json.load(f)
            if verbose:
                print(f"[OK] Loaded summary from {self.summary_file}")
        else:
            if verbose:
                print(f"[WARN] Summary file not found at {self.summary_file}")
        
        # Try to load target_std from multiple sources
        if self.summary is not None:
            # First try: from summary itself
            self.target_std = self.summary.get("target_std")
            
            # Second try: from best trial user_attrs
            if self.target_std is None:
                best_trial = self.summary.get("best_trial", {})
                user_attrs = best_trial.get("user_attrs", {})
                self.target_std = user_attrs.get("target_std")
            
            # Third try: from normalization stats in optuna results directory
            if self.target_std is None:
                norm_stats_file = self.study_dir / "normalization_stats.json"
                if norm_stats_file.exists():
                    try:
                        with open(norm_stats_file, 'r') as f:
                            norm_stats = json.load(f)
                            self.target_std = norm_stats.get("target_std")
                    except:
                        pass
            
            # Fourth try: from parent results directory (where train.py saves it)
            if self.target_std is None:
                parent_norm_file = Path("./results/normalization_stats.json")
                if parent_norm_file.exists():
                    try:
                        with open(parent_norm_file, 'r') as f:
                            norm_stats = json.load(f)
                            self.target_std = norm_stats.get("target_std")
                            if self.verbose and self.target_std:
                                print(f"[INFO] Using target_std from {parent_norm_file}")
                    except:
                        pass
        
        # Load study database if available
        if self.db_file.exists():
            try:
                storage_url = f"sqlite:///{self.db_file}"
                self.study = optuna.load_study(
                    study_name=self.summary.get("best_trial", {}).get("study_name", "tune_mlp"),
                    storage=storage_url,
                    load_if_exists=True,
                )
                if verbose:
                    print(f"[OK] Loaded Optuna study from database")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Could not load study database: {e}")
    
    def display_summary(self) -> None:
        """Display tuning summary."""
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return
        
        print(f"\n{'='*80}")
        print(f"TUNING SUMMARY")
        print(f"{'='*80}")
        
        summary = self.summary
        best = summary.get("best_trial", {})
        metrics = best.get("user_attrs", {})
        hyperparams = best.get("params", {})
        
        print(f"\nArchitecture: {summary.get('architecture', 'N/A').upper()}")
        print(f"Number of Trials: {summary.get('n_trials', 'N/A')}")
        print(f"Max Epochs per Trial: {summary.get('max_epochs', 'N/A')}")
        print(f"Total Trials Run: {summary.get('trials_count', 'N/A')}")
        print(f"Completed Trials: {summary.get('completed_trials', 'N/A')}")
        print(f"Optimization Time: {summary.get('optimization_time', 'N/A')}")
        
        print(f"\n{'-'*80}")
        print(f"BEST TRIAL: {best.get('number', 'N/A')}")
        print(f"{'-'*80}")
        print(f"\nMetrics:")
        
        val_rmse = best.get('value', float('nan'))
        test_rmse = metrics.get('test_rmse', float('nan'))
        test_rmse_denorm = metrics.get('test_rmse_denorm', float('nan'))
        test_mae = metrics.get('test_mae', float('nan'))
        
        print(f"  - Validation RMSE (norm): {val_rmse:.6f}")
        print(f"  - Test RMSE (norm): {test_rmse:.6f}")
        print(f"  - Test RMSE (denorm): {test_rmse_denorm:.2f} kg/m³")
        print(f"  - Test MAE (denorm): {test_mae:.2f} kg/m³")
        
        # Calculate accuracy metrics
        if self.target_std is not None and not np.isnan(test_rmse_denorm):
            # Accuracy relative to standard deviation
            # Formula: (1 - RMSE/std) * 100
            relative_accuracy = (1.0 - test_rmse_denorm / self.target_std) * 100.0
            
            # RMSE as percentage of std
            rmse_pct_of_std = (test_rmse_denorm / self.target_std) * 100.0
            
            print(f"\nAccuracy Metrics:")
            print(f"  - Relative Accuracy: {relative_accuracy:.2f}%")
            print(f"  - RMSE as % of Std Dev: {rmse_pct_of_std:.2f}%")
            print(f"  - Dataset Std Dev: {self.target_std:.2f} kg/m³")
            print(f"\n  Info: The model's typical prediction error ({test_rmse_denorm:.2f} kg/m³)")
            print(f"        is {rmse_pct_of_std:.2f}% of the dataset variability ({self.target_std:.2f} kg/m³).")
            print(f"        This represents {relative_accuracy:.2f}% relative accuracy.")
        
        print(f"\nTop Hyperparameters:")
        for i, (key, value) in enumerate(sorted(hyperparams.items())):
            if i < 15:
                print(f"  - {key}: {value}")
            else:
                break
        
        if len(hyperparams) > 15:
            print(f"  ... and {len(hyperparams) - 15} more hyperparameters")
        
        print(f"\n{'='*80}\n")
    
    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get best hyperparameters from tuning.
        
        Returns:
            Dictionary of best hyperparameters
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return {}
        
        best = self.summary.get("best_trial", {})
        return best.get("params", {})
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from tuning.
        
        Returns:
            Dictionary of best metrics
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return {}
        
        best = self.summary.get("best_trial", {})
        metrics = {
            "val_rmse": best.get("value"),
        }
        metrics.update(best.get("user_attrs", {}))
        return metrics
    
    def export_best_config(self, output_file: str = None) -> str:
        """Export best configuration to file.
        
        Args:
            output_file: Output filename (default: best_config.json)
            
        Returns:
            Path to exported file
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        best_hyperparams = self.get_best_hyperparameters()
        architecture = self.summary.get("architecture", "unknown")
        
        if output_file is None:
            output_file = "best_config.json"
        
        config_data = {
            "architecture": architecture,
            "hyperparameters": best_hyperparams,
            "metrics": self.get_best_metrics(),
            "tuning_info": {
                "n_trials": self.summary.get("n_trials"),
                "max_epochs": self.summary.get("max_epochs"),
                "total_trials": self.summary.get("trials_count"),
                "best_trial_number": self.summary.get("best_trial", {}).get("number"),
            },
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"✓ Best configuration exported to {output_path}")
        
        return str(output_path)
    
    def export_best_hyperparameters_for_training(self, output_file: str = None) -> str:
        """Export best hyperparameters in format for train.py.
        
        Args:
            output_file: Output filename (default: best_hyperparams.json)
            
        Returns:
            Path to exported file
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        best_hyperparams = self.get_best_hyperparameters()
        architecture = self.summary.get("architecture", "unknown")
        
        if output_file is None:
            output_file = "best_hyperparams.json"
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(best_hyperparams, f, indent=2, default=str)
        
        if self.verbose:
            print(f"✓ Best hyperparameters exported to {output_path}")
            print(f"\nYou can use these hyperparameters with train.py:")
            
            # Generate command line arguments
            cmd_parts = [f"python train.py --architecture {architecture}"]
            for key, value in best_hyperparams.items():
                # Convert underscore-separated keys to command line format
                arg_name = key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        cmd_parts.append(f"--{arg_name}")
                else:
                    cmd_parts.append(f"--{arg_name} {value}")
            
            print("\n  " + " \\\n  ".join(cmd_parts))
        
        return str(output_path)
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive text report.
        
        Args:
            output_file: Output filename (default: optuna_report.txt)
            
        Returns:
            Path to report file
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        if output_file is None:
            output_file = "optuna_report.txt"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPTUNA HYPERPARAMETER TUNING REPORT\n")
            f.write("="*80 + "\n\n")
            
            summary = self.summary
            best = summary.get("best_trial", {})
            metrics = best.get("user_attrs", {})
            hyperparams = best.get("params", {})
            
            # Summary section
            f.write("TUNING SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Architecture: {summary.get('architecture', 'N/A').upper()}\n")
            f.write(f"Number of Trials: {summary.get('n_trials', 'N/A')}\n")
            f.write(f"Max Epochs per Trial: {summary.get('max_epochs', 'N/A')}\n")
            f.write(f"Total Trials Run: {summary.get('trials_count', 'N/A')}\n")
            f.write(f"Completed Trials: {summary.get('completed_trials', 'N/A')}\n")
            f.write(f"Optimization Time: {summary.get('optimization_time', 'N/A')}\n\n")
            
            # Best trial section
            f.write("BEST TRIAL\n")
            f.write("-"*80 + "\n")
            f.write(f"Trial Number: {best.get('number', 'N/A')}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  • Validation RMSE (normalized): {best.get('value', float('nan')):.6f}\n")
            f.write(f"  • Test RMSE (normalized): {metrics.get('test_rmse', float('nan')):.6f}\n")
            f.write(f"  • Test RMSE (denormalized): {metrics.get('test_rmse_denorm', float('nan')):.2f} kg/m³\n")
            f.write(f"  • Test MAE (denormalized): {metrics.get('test_mae', float('nan')):.2f} kg/m³\n\n")
            
            f.write("Best Hyperparameters:\n")
            for key, value in sorted(hyperparams.items()):
                f.write(f"  • {key}: {value}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        if self.verbose:
            print(f"✓ Report generated to {output_path}")
        
        return str(output_path)
    
    def plot_results(self, output_file: str = None) -> Optional[str]:
        """Generate visualization plots of tuning results.
        
        Args:
            output_file: Output filename (default: optuna_results.png)
            
        Returns:
            Path to plot file, or None if matplotlib not available
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: matplotlib not available, skipping plots")
            return None
        
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        if output_file is None:
            output_file = "optuna_results.png"
        
        # Create figure with summary information
        fig = plt.figure(figsize=(12, 8))
        
        best = self.summary.get("best_trial", {})
        metrics = best.get("user_attrs", {})
        hyperparams = best.get("params", {})
        
        # Main plot - metrics summary
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        title_text = f"OPTUNA RESULTS - {self.summary.get('architecture', 'UNKNOWN').upper()}"
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # Summary text
        summary_text = f"""
TUNING SUMMARY
{'─'*70}
Total Trials: {self.summary.get('trials_count', 'N/A')}
Max Epochs per Trial: {self.summary.get('max_epochs', 'N/A')}

BEST TRIAL: {best.get('number', 'N/A')}
{'─'*70}
Validation RMSE: {best.get('value', float('nan')):.6f}
Test RMSE (denorm): {metrics.get('test_rmse_denorm', float('nan')):.2f} kg/m³
Test MAE (denorm): {metrics.get('test_mae', float('nan')):.2f} kg/m³

TOP HYPERPARAMETERS
{'─'*70}
"""
        
        # Add top hyperparameters
        for i, (key, value) in enumerate(list(sorted(hyperparams.items()))[:15]):
            if i < 15:
                summary_text += f"{key}: {value}\n"
        
        if len(hyperparams) > 15:
            summary_text += f"... and {len(hyperparams) - 15} more hyperparameters"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"✓ Plot saved to {output_file}")
        
        plt.close()
        return str(output_file)
    
    def plot_optimization_history(self, output_file: str = None) -> Optional[str]:
        """Plot optimization history (if study is available).
        
        Args:
            output_file: Output filename
            
        Returns:
            Path to plot file, or None
        """
        if not HAS_MATPLOTLIB or self.study is None:
            return None
        
        if output_file is None:
            output_file = "optuna_history.png"
        
        try:
            fig = optuna.visualization.plot_optimization_history(self.study).to_html()
            # Save as HTML instead of PNG for better interactivity
            html_file = output_file.replace('.png', '.html')
            with open(html_file, 'w') as f:
                f.write(fig)
            if self.verbose:
                print(f"✓ Optimization history saved to {html_file}")
            return html_file
        except Exception as e:
            if self.verbose:
                print(f"⚠ Could not generate optimization history: {e}")
            return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Optuna hyperparameter tuning results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing Optuna results (default: find latest)",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        default=True,
        help="Display tuning summary (default: True)",
    )
    parser.add_argument(
        "--export-config",
        action="store_true",
        help="Export best configuration",
    )
    parser.add_argument(
        "--export-hyperparams",
        action="store_true",
        help="Export best hyperparameters for train.py",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate text report",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export and generate all outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for exports (default: current directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    
    return parser.parse_args()


def find_latest_results_dir() -> Optional[Path]:
    """Find latest optuna_results directory.
    
    Returns:
        Path to latest results directory, or None if not found
    """
    optuna_dirs = sorted(Path(".").glob("optuna_results_*"))
    if not optuna_dirs:
        return None
    
    # Get latest (most recent) directory
    latest = sorted(optuna_dirs, key=lambda p: p.stat().st_mtime)[-1]
    return latest


def main():
    """Main analysis script."""
    args = parse_arguments()
    
    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = find_latest_results_dir()
    
    if results_dir is None or not results_dir.exists():
        print("ERROR: Results directory not found!")
        print("Please specify with --results-dir or ensure Optuna results exist")
        return
    
    # Create analyzer
    analyzer = OptunaResultsAnalyzer(str(results_dir), verbose=args.verbose)
    
    # Display summary
    if args.show_summary:
        analyzer.display_summary()
    
    # Export and generate
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.export_config or args.all:
        analyzer.export_best_config(str(output_dir / "best_config.json"))
    
    if args.export_hyperparams or args.all:
        analyzer.export_best_hyperparameters_for_training(
            str(output_dir / "best_hyperparams.json")
        )
    
    if args.generate_report or args.all:
        analyzer.generate_report(str(output_dir / "optuna_report.txt"))
    
    if args.plot or args.all:
        analyzer.plot_results(str(output_dir / "optuna_results.png"))


if __name__ == "__main__":
    main()
