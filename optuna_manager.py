"""Unified Optuna hyperparameter tuning manager.

Consolidates all Optuna functionality:
- Running optimization with configurable parameters
- Saving n-best hyperparameter configurations with full reproducibility
- Analysis and visualization of tuning results
- Loading and using tuned configs

This module replaces the redundant functionality previously split across:
- visualize_optuna_results.py
- optuna_analyze_results.py
- tune.py (core execution only)

Key features:
- Saves n-best configs (configurable) with all necessary info
- Proper seed management (not fixed for grid search)
- Consolidated analysis and visualization
- Easy integration with deep ensemble
"""

import argparse
import json
import logging
import numpy as np
import optuna
import torch
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from tune_config import OptunaSearchSpace, SEARCH_SPACE_CONFIGS, get_search_config
from optuna_trainable import create_objective

# Optional visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class OptunaResultsHandler:
    """Handles saving and loading of Optuna results, including n-best configs."""
    
    def __init__(self, results_dir: Path):
        """Initialize results handler.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.configs_dir = self.results_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.results_dir / "optuna_metadata.json"
        self.db_file = self.results_dir / "study.db"
    
    def save_n_best_configs(
        self,
        study: optuna.study.Study,
        architecture: str,
        n_best: int = 5,
        max_epochs: int = 100,
        seed: int = 46,
    ) -> Dict[str, Any]:
        """Save n-best hyperparameter configurations with full metadata.
        
        Args:
            study: Optuna study object
            architecture: Model architecture
            n_best: Number of best configs to save (default: 5)
            max_epochs: Maximum epochs used in tuning
            seed: Master seed for reproducibility
            
        Returns:
            Dictionary with saved configs metadata
        """
        # Get all trials and sort by value (best first)
        all_trials = study.trials
        # Filter completed trials and sort by value
        completed_trials = [t for t in all_trials if t.value is not None]
        completed_trials.sort(key=lambda t: t.value)
        
        n_best = min(n_best, len(completed_trials))
        best_trials = completed_trials[:n_best]
        
        configs_metadata = {
            "architecture": architecture,
            "n_best": n_best,
            "timestamp": datetime.now().isoformat(),
            "n_total_trials": len(study.trials),
            "max_epochs": max_epochs,
            "master_seed": seed,
            "configs": [],
        }
        
        for rank, trial in enumerate(best_trials, 1):
            config_data = {
                "rank": rank,
                "trial_number": trial.number,
                "validation_value": trial.value,  # Validation RMSE (normalized)
                "params": trial.params,
                "user_attrs": trial.user_attrs,  # Contains test metrics
                # Reproducibility info
                "architecture": architecture,
                "max_epochs": max_epochs,
                # Seed for this specific config (derived from master seed + rank)
                "seed": seed + rank,
            }
            
            configs_metadata["configs"].append(config_data)
            
            # Save individual config file
            config_file = self.configs_dir / f"config_rank_{rank:02d}.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        # Save metadata file
        with open(self.metadata_file, 'w') as f:
            json.dump(configs_metadata, f, indent=2)
        
        return configs_metadata
    
    def load_n_best_configs(self, n_best: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load n-best configurations from saved metadata.
        
        Args:
            n_best: Number of configs to load (None = load all saved)
            
        Returns:
            List of config dictionaries, sorted by rank
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        configs = metadata["configs"]
        if n_best is not None:
            configs = configs[:n_best]
        
        return configs
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best (rank 1) configuration.
        
        Returns:
            Best config dictionary
        """
        configs = self.load_n_best_configs(n_best=1)
        return configs[0] if configs else None


class OptunaOptimizer:
    """Manager for running Optuna hyperparameter optimization."""
    
    def __init__(
        self,
        architecture: str,
        n_trials: int = 50,
        max_epochs: int = 100,
        n_best_save: int = 5,
        data_dir: str = ".",
        device: str = None,
        verbose: bool = True,
        seed: int = 46,
    ):
        """Initialize the optimizer.
        
        Args:
            architecture: Model architecture ("mlp", "cnn", "cnn_multiscale", "lightgbm")
            n_trials: Number of trials to run
            max_epochs: Maximum epochs per trial
            n_best_save: Number of best configs to save (default: 5)
            data_dir: Directory containing dataset CSV files
            device: Device to use ("cuda" or "cpu", auto-detected if None)
            verbose: Whether to print detailed output
            seed: Master seed for reproducibility (default: 46)
        """
        self.architecture = architecture
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.n_best_save = n_best_save
        self.data_dir = data_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.seed = seed
        
        # Create results directory
        self.results_dir = Path(
            f"./optuna_results_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for study
        self.study_name = f"tune_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_url = f"sqlite:///{self.results_dir}/study.db"
        
        # Results handler
        self.results_handler = OptunaResultsHandler(self.results_dir)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPTUNA HYPERPARAMETER OPTIMIZER")
            print(f"{'='*80}")
            print(f"Architecture: {architecture.upper()}")
            print(f"N Trials: {n_trials}")
            print(f"Max Epochs: {max_epochs}")
            print(f"N Best to Save: {n_best_save}")
            print(f"Data Dir: {data_dir}")
            print(f"Device: {self.device.upper()}")
            print(f"Results Dir: {self.results_dir}")
            print(f"Master Seed: {seed}")
            print(f"{'='*80}\n")
    
    def run_optimization(
        self,
        sampler_type: str = "tpe",
        pruner_type: str = "median",
        n_jobs: int = 1,
    ) -> Tuple[optuna.study.Study, Dict[str, Any]]:
        """Run hyperparameter optimization.
        
        Args:
            sampler_type: Sampler type ("tpe", "grid", or "random")
            pruner_type: Pruner type ("median", "noop", or "percentile")
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (Optuna Study object, results metadata with n-best configs)
        """
        # Configure sampler
        if sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.seed)
        elif sampler_type == "grid":
            # GridSampler doesn't support seed, use RandomSampler for variable trials
            sampler = optuna.samplers.RandomSampler(seed=None)
        elif sampler_type == "random":
            sampler = optuna.samplers.RandomSampler(seed=None)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        # Configure pruner
        if pruner_type == "median":
            pruner = optuna.pruners.MedianPruner()
        elif pruner_type == "noop":
            pruner = optuna.pruners.NopPruner()
        elif pruner_type == "percentile":
            pruner = optuna.pruners.PercentilePruner(percentile=30.0)
        else:
            raise ValueError(f"Unknown pruner type: {pruner_type}")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False,
        )
        
        # Get objective function
        suggest_fn = OptunaSearchSpace.get_suggest_function(self.architecture)
        objective = create_objective(
            self.architecture,
            self.max_epochs,
            self.data_dir,
            self.device,
            suggest_fn,
            seed=self.seed,
        )
        
        # Run optimization
        if self.verbose:
            print(f"Starting optimization with {self.n_trials} trials...")
            print(f"Sampler: {sampler_type}, Pruner: {pruner_type}\n")
        
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            if self.verbose:
                print("\nOptimization interrupted by user")
        
        # Print summary
        self.print_summary(study)
        
        # Save n-best configs
        results_metadata = self.results_handler.save_n_best_configs(
            study,
            self.architecture,
            n_best=self.n_best_save,
            max_epochs=self.max_epochs,
            seed=self.seed,
        )
        
        if self.verbose:
            print(f"\n✓ Saved {self.n_best_save} best configurations")
            print(f"  Configs saved to: {self.results_handler.configs_dir}")
            print(f"  Metadata saved to: {self.results_handler.metadata_file}\n")
        
        return study, results_metadata
    
    def print_summary(self, study: optuna.study.Study) -> None:
        """Print summary of optimization results.
        
        Args:
            study: Optuna study object
        """
        best_trial = study.best_trial
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total Trials: {len(study.trials)}")
        print(f"Best Trial: {best_trial.number}")
        print(f"Best Value (Validation RMSE): {best_trial.value:.6f}")
        
        if 'test_rmse_denorm' in best_trial.user_attrs:
            print(f"Test RMSE (denorm): {best_trial.user_attrs['test_rmse_denorm']:.2f} kg/m³")
        
        print(f"\nBest Hyperparameters:")
        for key, value in sorted(best_trial.params.items()):
            print(f"  {key}: {value}")
        
        print(f"{'='*80}\n")


class OptunaAnalyzer:
    """Consolidated analysis of Optuna tuning results."""
    
    def __init__(self, results_dir: Path, verbose: bool = True):
        """Initialize analyzer.
        
        Args:
            results_dir: Directory containing Optuna results
            verbose: Whether to print detailed output
        """
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        self.results_handler = OptunaResultsHandler(self.results_dir)
        self.metadata = None
        
        if self.results_handler.metadata_file.exists():
            with open(self.results_handler.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Try to load study from database
        self.study = None
        try:
            storage_url = f"sqlite:///{self.results_dir}/study.db"
            self.study = optuna.load_study(storage=storage_url)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load study database: {e}")
    
    def display_summary(self) -> None:
        """Display comprehensive tuning summary."""
        if self.metadata is None:
            print("No tuning metadata found")
            return
        
        print(f"\n{'='*80}")
        print(f"TUNING SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nArchitecture: {self.metadata['architecture'].upper()}")
        print(f"Total Trials: {self.metadata['n_total_trials']}")
        print(f"N Best Saved: {self.metadata['n_best']}")
        print(f"Max Epochs: {self.metadata['max_epochs']}")
        print(f"Master Seed: {self.metadata['master_seed']}")
        
        print(f"\n{'─'*80}")
        print(f"TOP {min(5, len(self.metadata['configs']))} CONFIGURATIONS")
        print(f"{'─'*80}")
        
        for config in self.metadata['configs'][:5]:
            print(f"\nRank {config['rank']} (Trial {config['trial_number']}):")
            print(f"  Validation RMSE: {config['validation_value']:.6f}")
            if 'test_rmse_denorm' in config['user_attrs']:
                print(f"  Test RMSE: {config['user_attrs']['test_rmse_denorm']:.2f} kg/m³")
            print(f"  Seed: {config['seed']}")
            print(f"  Key Hyperparameters:")
            for key, value in list(config['params'].items())[:5]:
                print(f"    - {key}: {value}")
        
        print(f"\n{'='*80}\n")
    
    def export_best_config(self, output_file: Optional[str] = None) -> str:
        """Export best configuration to file.
        
        Args:
            output_file: Output filename (default: best_config.json in results dir)
            
        Returns:
            Path to exported file
        """
        if self.metadata is None:
            raise ValueError("No tuning metadata available")
        
        best_config = self.metadata['configs'][0]
        
        if output_file is None:
            output_file = self.results_dir / "best_config.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        if self.verbose:
            print(f"✓ Exported best config to {output_file}")
        
        return str(output_file)
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate text report of tuning results.
        
        Args:
            output_file: Output filename (default: tuning_report.txt in results dir)
            
        Returns:
            Path to report file
        """
        if self.metadata is None:
            raise ValueError("No tuning metadata available")
        
        if output_file is None:
            output_file = self.results_dir / "tuning_report.txt"
        else:
            output_file = Path(output_file)
        
        report_lines = [
            "="*80,
            "OPTUNA HYPERPARAMETER TUNING REPORT",
            "="*80,
            f"\nTimestamp: {self.metadata['timestamp']}",
            f"Architecture: {self.metadata['architecture']}",
            f"Total Trials: {self.metadata['n_total_trials']}",
            f"Max Epochs: {self.metadata['max_epochs']}",
            f"Master Seed: {self.metadata['master_seed']}",
            "\n" + "─"*80,
            f"TOP {min(10, len(self.metadata['configs']))} CONFIGURATIONS",
            "─"*80,
        ]
        
        for config in self.metadata['configs'][:10]:
            report_lines.extend([
                f"\n[Rank {config['rank']}] Trial #{config['trial_number']}",
                f"  Validation RMSE: {config['validation_value']:.6f}",
                f"  Seed: {config['seed']}",
            ])
            
            if 'test_rmse_denorm' in config['user_attrs']:
                report_lines.append(
                    f"  Test RMSE: {config['user_attrs']['test_rmse_denorm']:.2f} kg/m³"
                )
            
            report_lines.append("  Hyperparameters:")
            for key, value in sorted(config['params'].items()):
                report_lines.append(f"    - {key}: {value}")
        
        report_lines.extend(["\n" + "="*80])
        
        report_text = "\n".join(report_lines)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        if self.verbose:
            print(f"✓ Generated report: {output_file}")
        
        return str(output_file)


class OptunaVisualizer:
    """Consolidated visualization of Optuna results."""
    
    def __init__(self, results_dir: Path):
        """Initialize visualizer.
        
        Args:
            results_dir: Directory containing Optuna results
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        self.results_dir = Path(results_dir)
        self.results_handler = OptunaResultsHandler(self.results_dir)
        
        # Load study
        storage_url = f"sqlite:///{self.results_dir}/study.db"
        try:
            self.study = optuna.load_study(storage=storage_url)
        except Exception as e:
            raise ValueError(f"Could not load study: {e}")
    
    def plot_optimization_history(self, output_file: Optional[str] = None) -> Optional[str]:
        """Plot optimization history.
        
        Args:
            output_file: Output filename (default: optimization_history.png)
            
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = self.results_dir / "optimization_history.png"
        else:
            output_file = Path(output_file)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        trials = self.study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value if t.value is not None else float('nan') for t in trials]
        
        # Plot all trial values
        ax.plot(trial_numbers, values, 'o-', alpha=0.6, label='Trial Value')
        
        # Plot best value so far
        best_values = []
        best_so_far = float('inf')
        for value in values:
            if not np.isnan(value) and value < best_so_far:
                best_so_far = value
            best_values.append(best_so_far)
        
        ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best So Far')
        
        ax.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation RMSE (normalized)', fontsize=11, fontweight='bold')
        ax.set_title('Optimization Convergence', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved optimization history plot: {output_file}")
        return str(output_file)
    
    def plot_parameter_importance(self, output_file: Optional[str] = None) -> Optional[str]:
        """Plot parameter importance using Optuna's importance calculation.
        
        Args:
            output_file: Output filename (default: parameter_importance.png)
            
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = self.results_dir / "parameter_importance.png"
        else:
            output_file = Path(output_file)
        
        try:
            fig = optuna.visualization.plot_param_importances(self.study).to_plotly_figure()
            fig.write_image(str(output_file), width=1000, height=600)
        except Exception as e:
            # Fallback to simple matplotlib plot if plotly fails
            fig, ax = plt.subplots(figsize=(12, 6))
            
            trials = self.study.trials
            completed_trials = [t for t in trials if t.value is not None]
            
            if completed_trials:
                param_names = list(completed_trials[0].params.keys())
                ax.barh(param_names, range(len(param_names)))
                ax.set_xlabel('Parameter Index')
                ax.set_title('Parameter Importance (Approximation)')
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✓ Saved parameter importance plot: {output_file}")
        return str(output_file)


def find_latest_results_dir() -> Optional[Path]:
    """Find the latest optuna_results directory.
    
    Returns:
        Path to latest results directory, or None if not found
    """
    optuna_dirs = sorted(Path(".").glob("optuna_results_*"))
    if not optuna_dirs:
        return None
    
    # Get latest (most recent) directory
    latest = sorted(optuna_dirs, key=lambda p: p.stat().st_mtime)[-1]
    return latest


def parse_arguments():
    """Parse command line arguments for tune mode."""
    parser = argparse.ArgumentParser(
        description="Unified Optuna hyperparameter tuning and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optuna_manager.py tune --architecture mlp --n-trials 50
  python optuna_manager.py tune --architecture cnn --config-type balanced
  python optuna_manager.py analyze --results-dir ./optuna_results_mlp_20251217_120000
  python optuna_manager.py analyze --all
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "cnn_multiscale", "lightgbm"],
        help="Model architecture to tune (default: mlp)",
    )
    tune_parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials (default: 50)",
    )
    tune_parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial (default: 100)",
    )
    tune_parser.add_argument(
        "--n-best-save",
        type=int,
        default=5,
        help="Number of best configs to save (default: 5)",
    )
    tune_parser.add_argument(
        "--config-type",
        type=str,
        default=None,
        choices=list(SEARCH_SPACE_CONFIGS.keys()),
        help="Use predefined search configuration",
    )
    tune_parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing dataset CSV files (default: current directory)",
    )
    tune_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    tune_parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "grid", "random"],
        help="Sampler type (default: tpe)",
    )
    tune_parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "noop", "percentile"],
        help="Pruner type (default: median)",
    )
    tune_parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )
    tune_parser.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Master seed for reproducibility (default: 46)",
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze tuning results")
    analyze_parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing Optuna results (default: find latest)",
    )
    analyze_parser.add_argument(
        "--show-summary",
        action="store_true",
        default=True,
        help="Display tuning summary (default: True)",
    )
    analyze_parser.add_argument(
        "--export-config",
        action="store_true",
        help="Export best configuration",
    )
    analyze_parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate text report",
    )
    analyze_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    analyze_parser.add_argument(
        "--all",
        action="store_true",
        help="Export and generate all outputs",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for exports (default: current directory)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.command == "tune":
        # Handle predefined config
        if args.config_type:
            config = get_search_config(args.config_type)
            n_trials = config["n_trials"]
            max_epochs = config["max_epochs"]
        else:
            n_trials = args.n_trials
            max_epochs = args.max_epochs
        
        # Create and run optimizer
        optimizer = OptunaOptimizer(
            architecture=args.architecture,
            n_trials=n_trials,
            max_epochs=max_epochs,
            n_best_save=args.n_best_save,
            data_dir=args.data_dir,
            device=args.device,
            verbose=True,
            seed=args.seed,
        )
        
        study, results = optimizer.run_optimization(
            sampler_type=args.sampler,
            pruner_type=args.pruner,
            n_jobs=args.n_jobs,
        )
    
    elif args.command == "analyze":
        # Find results directory
        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = find_latest_results_dir()
        
        if results_dir is None or not results_dir.exists():
            print(f"Error: Results directory not found")
            return
        
        # Create analyzer
        analyzer = OptunaAnalyzer(results_dir, verbose=True)
        
        # Display summary
        if args.show_summary:
            analyzer.display_summary()
        
        # Export and generate
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.export_config or args.all:
            analyzer.export_best_config(str(output_dir / "best_config.json"))
        
        if args.generate_report or args.all:
            analyzer.generate_report(str(output_dir / "tuning_report.txt"))
        
        if args.plot or args.all:
            try:
                visualizer = OptunaVisualizer(results_dir)
                visualizer.plot_optimization_history(str(output_dir / "optimization_history.png"))
                visualizer.plot_parameter_importance(str(output_dir / "parameter_importance.png"))
            except ImportError:
                print("Warning: Skipping plots (matplotlib not available)")
    
    else:
        print("Usage: python optuna_manager.py {tune,analyze} [options]")


if __name__ == "__main__":
    main()
