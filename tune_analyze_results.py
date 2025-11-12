"""Analysis and visualization of Ray Tune hyperparameter tuning results.

This script provides tools to:
- Load and analyze tuning results
- Compare different trials
- Visualize hyperparameter importance
- Generate performance reports
- Extract and export best configurations
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from ray.tune import ExperimentAnalysis
    HAS_RAY_TUNE = True
except ImportError:
    HAS_RAY_TUNE = False


class TuneResultsAnalyzer:
    """Analyzer for Ray Tune hyperparameter tuning results."""
    
    def __init__(self, results_dir: str, verbose: bool = True):
        """Initialize the analyzer.
        
        Args:
            results_dir: Directory containing Ray Tune results (tune_results_*/*)
            verbose: Whether to print detailed output
        """
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        self.summary_file = self.results_dir / "tune_summary.json"
        self.summary = None
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"RAY TUNE RESULTS ANALYZER")
            print(f"{'='*80}")
            print(f"Results Directory: {self.results_dir}")
        
        # Load summary
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.summary = json.load(f)
            if verbose:
                print(f"✓ Loaded summary from {self.summary_file}")
        else:
            if verbose:
                print(f"⚠ Summary file not found at {self.summary_file}")
    
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
        metrics = best.get("metrics", {})
        hyperparams = best.get("hyperparameters", {})
        
        print(f"\nArchitecture: {summary.get('architecture', 'N/A').upper()}")
        print(f"Number of Samples: {summary.get('num_samples', 'N/A')}")
        print(f"Max Epochs: {summary.get('max_epochs', 'N/A')}")
        print(f"Total Trials: {summary.get('trials_count', 'N/A')}")
        print(f"Search Time: {summary.get('search_time', 'N/A')}")
        
        print(f"\n{'─'*80}")
        print(f"BEST TRIAL: {best.get('trial_id', 'N/A')}")
        print(f"{'─'*80}")
        print(f"\nMetrics:")
        print(f"  • Validation RMSE (norm): {metrics.get('val_rmse', float('nan')):.6f}")
        print(f"  • Train RMSE (norm): {metrics.get('train_rmse', float('nan')):.6f}")
        print(f"  • Test RMSE (norm): {metrics.get('test_rmse', float('nan')):.6f}")
        print(f"  • Test RMSE (denorm): {metrics.get('test_rmse_denorm', float('nan')):.2f} kg/m³")
        
        print(f"\nTop Hyperparameters:")
        for key, value in sorted(hyperparams.items())[:10]:
            print(f"  • {key}: {value}")
        
        if len(hyperparams) > 10:
            print(f"  ... and {len(hyperparams) - 10} more hyperparameters")
        
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
        return best.get("hyperparameters", {})
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from tuning.
        
        Returns:
            Dictionary of best metrics
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return {}
        
        best = self.summary.get("best_trial", {})
        return best.get("metrics", {})
    
    def export_best_config(self, output_file: str = None) -> str:
        """Export best configuration to file.
        
        Args:
            output_file: Output filename (default: best_config_<architecture>.json)
            
        Returns:
            Path to exported file
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        best_hyperparams = self.get_best_hyperparameters()
        architecture = self.summary.get("architecture", "unknown")
        
        if output_file is None:
            output_file = f"best_config_{architecture}.json"
        
        config_data = {
            "architecture": architecture,
            "hyperparameters": best_hyperparams,
            "metrics": self.get_best_metrics(),
            "tuning_info": {
                "num_samples": self.summary.get("num_samples"),
                "max_epochs": self.summary.get("max_epochs"),
                "total_trials": self.summary.get("trials_count"),
                "best_trial_id": self.summary.get("best_trial", {}).get("trial_id"),
            },
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        if self.verbose:
            print(f"✓ Best configuration exported to {output_path}")
        
        return str(output_path)
    
    def export_best_hyperparameters_for_training(self, output_file: str = None) -> str:
        """Export best hyperparameters in format for train.py.
        
        Args:
            output_file: Output filename (default: best_hyperparams_<architecture>.json)
            
        Returns:
            Path to exported file
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        best_hyperparams = self.get_best_hyperparameters()
        architecture = self.summary.get("architecture", "unknown")
        
        if output_file is None:
            output_file = f"best_hyperparams_{architecture}.json"
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(best_hyperparams, f, indent=2)
        
        if self.verbose:
            print(f"✓ Best hyperparameters exported to {output_path}")
            print(f"\nYou can use these hyperparameters with train.py:")
            
            # Generate command line arguments
            cmd_parts = [f"python train.py --architecture {architecture}"]
            for key, value in best_hyperparams.items():
                # Convert hyphen-separated keys to command line format
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
            output_file: Output filename (default: tune_report.txt)
            
        Returns:
            Path to report file
        """
        if self.summary is None:
            print("ERROR: Summary not loaded")
            return None
        
        if output_file is None:
            output_file = "tune_report.txt"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RAY TUNE HYPERPARAMETER TUNING REPORT\n")
            f.write("="*80 + "\n\n")
            
            summary = self.summary
            best = summary.get("best_trial", {})
            metrics = best.get("metrics", {})
            hyperparams = best.get("hyperparameters", {})
            
            # Summary section
            f.write("TUNING SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Architecture: {summary.get('architecture', 'N/A').upper()}\n")
            f.write(f"Number of Samples: {summary.get('num_samples', 'N/A')}\n")
            f.write(f"Max Epochs: {summary.get('max_epochs', 'N/A')}\n")
            f.write(f"Total Trials: {summary.get('trials_count', 'N/A')}\n")
            f.write(f"Search Time: {summary.get('search_time', 'N/A')}\n\n")
            
            # Best trial section
            f.write("BEST TRIAL\n")
            f.write("-"*80 + "\n")
            f.write(f"Trial ID: {best.get('trial_id', 'N/A')}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  • Validation RMSE (normalized): {metrics.get('val_rmse', float('nan')):.6f}\n")
            f.write(f"  • Training RMSE (normalized): {metrics.get('train_rmse', float('nan')):.6f}\n")
            f.write(f"  • Test RMSE (normalized): {metrics.get('test_rmse', float('nan')):.6f}\n")
            f.write(f"  • Test RMSE (denormalized): {metrics.get('test_rmse_denorm', float('nan')):.2f} kg/m³\n\n")
            
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
            output_file: Output filename (default: tune_results.png)
            
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
            output_file = "tune_results.png"
        
        # Create figure with summary information
        fig = plt.figure(figsize=(12, 8))
        
        best = self.summary.get("best_trial", {})
        metrics = best.get("metrics", {})
        hyperparams = best.get("hyperparameters", {})
        
        # Main plot - metrics summary
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        title_text = f"RAY TUNE RESULTS - {self.summary.get('architecture', 'UNKNOWN').upper()}"
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # Summary text
        summary_text = f"""
TUNING SUMMARY
{'─'*70}
Total Trials: {self.summary.get('trials_count', 'N/A')}
Max Epochs per Trial: {self.summary.get('max_epochs', 'N/A')}

BEST TRIAL: {best.get('trial_id', 'N/A')}
{'─'*70}
Validation RMSE: {metrics.get('val_rmse', float('nan')):.6f}
Training RMSE: {metrics.get('train_rmse', float('nan')):.6f}
Test RMSE: {metrics.get('test_rmse', float('nan')):.6f}
Test RMSE (denorm): {metrics.get('test_rmse_denorm', float('nan')):.2f} kg/m³

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Ray Tune hyperparameter tuning results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing tune results (default: find latest)",
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
    """Find latest tune_results directory.
    
    Returns:
        Path to latest results directory, or None if not found
    """
    tune_dirs = sorted(Path(".").glob("tune_results_*"))
    if not tune_dirs:
        return None
    
    # Get latest (most recent) directory
    latest = sorted(tune_dirs, key=lambda p: p.stat().st_mtime)[-1]
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
        print("Please specify with --results-dir or ensure tune results exist")
        return
    
    # Create analyzer
    analyzer = TuneResultsAnalyzer(str(results_dir), verbose=args.verbose)
    
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
        analyzer.generate_report(str(output_dir / "tune_report.txt"))
    
    if args.plot or args.all:
        analyzer.plot_results(str(output_dir / "tune_results.png"))


if __name__ == "__main__":
    main()
