"""Main hyperparameter tuning script using Optuna.

This script runs hyperparameter optimization for any of the 3 model architectures:
- mlp: Multi-Layer Perceptron
- cnn: Convolutional Neural Network
- cnn_multiscale: Multi-Scale CNN

Usage:
    python tune.py --architecture mlp --n-trials 50 --max-epochs 100
    python tune.py --architecture cnn --config-type balanced
"""

import argparse
import torch
import json
import optuna
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from tune_config import OptunaSearchSpace, SEARCH_SPACE_CONFIGS, get_search_config
from optuna_trainable import create_objective


class OptunaOptimizer:
    """Manager for Optuna hyperparameter optimization."""
    
    def __init__(
        self,
        architecture: str,
        n_trials: int = 50,
        max_epochs: int = 100,
        data_dir: str = ".",
        device: str = None,
        verbose: bool = True,
        seed: int = 46,
    ):
        """Initialize the optimizer.
        
        Args:
            architecture: Model architecture ("mlp", "cnn", "cnn_multiscale", "gnn")
            n_trials: Number of trials to run
            max_epochs: Maximum epochs per trial
            data_dir: Directory containing dataset CSV files
            device: Device to use ("cuda" or "cpu", auto-detected if None)
            verbose: Whether to print detailed output
            seed: Random seed for reproducibility (default: 46)
        """
        self.architecture = architecture
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.data_dir = data_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.seed = seed
        
        # Create results directory
        self.results_dir = Path(f"./optuna_results_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for study
        self.study_name = f"tune_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_url = f"sqlite:///{self.results_dir}/study.db"
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPTUNA HYPERPARAMETER OPTIMIZER")
            print(f"{'='*80}")
            print(f"Architecture: {architecture.upper()}")
            print(f"N Trials: {n_trials}")
            print(f"Max Epochs: {max_epochs}")
            print(f"Data Dir: {data_dir}")
            print(f"Device: {self.device.upper()}")
            print(f"Results Dir: {self.results_dir}")
            print(f"Study Name: {self.study_name}")
            print(f"{'='*80}\n")
    
    def run_optimization(
        self,
        sampler_type: str = "tpe",
        pruner_type: str = "median",
        n_jobs: int = 1,
    ) -> optuna.study.Study:
        """Run hyperparameter optimization.
        
        Args:
            sampler_type: Sampler type ("tpe", "grid", or "random")
            pruner_type: Pruner type ("median", "noop", or "percentile")
            n_jobs: Number of parallel jobs
            
        Returns:
            Optuna Study object
        """
        # Configure sampler
        if sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampler_type == "grid":
            sampler = optuna.samplers.GridSampler()
        elif sampler_type == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
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
        
        # Save results
        self.save_results_summary(study)
        
        if self.verbose:
            self.print_summary(study)
        
        return study
    
    def print_summary(self, study: optuna.study.Study) -> None:
        """Print optimization summary.
        
        Args:
            study: Optuna Study object
        """
        best_trial = study.best_trial
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Best Trial Number: {best_trial.number}")
        print(f"Best Validation RMSE (normalized): {best_trial.value:.6f}")
        
        # Get denormalized values
        target_std = best_trial.user_attrs.get('target_std', None)
        val_rmse_denorm = best_trial.user_attrs.get('val_rmse_denorm', None)
        test_rmse_denorm = best_trial.user_attrs.get('test_rmse_denorm', None)
        
        if val_rmse_denorm is not None and not (isinstance(val_rmse_denorm, float) and val_rmse_denorm == float('inf')):
            print(f"Best Validation RMSE (denorm): {val_rmse_denorm:.2f} kg/m³")
        else:
            print(f"Best Validation RMSE (denorm): N/A")
        
        if test_rmse_denorm is not None and not (isinstance(test_rmse_denorm, float) and test_rmse_denorm == float('inf')):
            print(f"Best Test RMSE (denorm): {test_rmse_denorm:.2f} kg/m³")
        else:
            print(f"Best Test RMSE (denorm): N/A")
        
        if target_std is not None:
            print(f"Target Std Dev: {target_std:.2f} kg/m³")
        
        print(f"Total Trials: {len(study.trials)}")
        print(f"Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        print(f"\nBest Hyperparameters:")
        for key, value in sorted(best_trial.params.items()):
            print(f"  - {key}: {value}")
        
        print(f"\n{'='*80}\n")
    
    def save_results_summary(self, study: optuna.study.Study) -> None:
        """Save optimization results summary.
        
        Args:
            study: Optuna Study object
        """
        best_trial = study.best_trial
        
        # Extract target_std from best trial if available
        target_std = best_trial.user_attrs.get("target_std", None)
        
        summary = {
            "architecture": self.architecture,
            "n_trials": self.n_trials,
            "max_epochs": self.max_epochs,
            "optimization_time": datetime.now().isoformat(),
            "target_std": target_std,  # Store for accuracy calculation
            "best_trial": {
                "number": best_trial.number,
                "value": float(best_trial.value),
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs,
            },
            "trials_count": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
        
        # Save to JSON
        summary_file = self.results_dir / "optuna_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.verbose:
            print(f"[OK] Summary saved to: {summary_file}")
    
    def get_best_hyperparameters(self, study: optuna.study.Study) -> Dict[str, Any]:
        """Get best hyperparameters from study.
        
        Args:
            study: Optuna Study object
            
        Returns:
            Dictionary of best hyperparameters
        """
        return study.best_trial.params
    
    def export_best_config(self, study: optuna.study.Study, output_file: str = None) -> str:
        """Export best configuration to file.
        
        Args:
            study: Optuna Study object
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.results_dir / "best_config.json"
        else:
            output_file = Path(output_file)
        
        best_trial = study.best_trial
        config_data = {
            "architecture": self.architecture,
            "hyperparameters": best_trial.params,
            "metrics": {
                "val_rmse": float(best_trial.value),
                "test_rmse": float(best_trial.user_attrs.get("test_rmse", float('nan'))),
                "test_rmse_denorm": float(best_trial.user_attrs.get("test_rmse_denorm", float('nan'))),
                "test_mae": float(best_trial.user_attrs.get("test_mae", float('nan'))),
            },
            "trial_info": {
                "best_trial_number": best_trial.number,
                "n_trials": self.n_trials,
                "max_epochs": self.max_epochs,
            },
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"✓ Best configuration exported to {output_file}")
        
        return str(output_file)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune.py --architecture mlp --n-trials 50
  python tune.py --architecture cnn --config-type balanced
  python tune.py --architecture gnn --n-trials 100 --sampler tpe
        """
    )
    
    # Architecture selection
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "cnn_multiscale", "lightgbm"],
        help="Model architecture to tune (default: mlp)",
    )
    
    # Optimization parameters
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials (default: 50)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial (default: 100)",
    )
    
    # Config type (alternative to n-trials/max-epochs)
    parser.add_argument(
        "--config-type",
        type=str,
        default=None,
        choices=list(SEARCH_SPACE_CONFIGS.keys()),
        help="Use predefined search configuration",
    )
    
    # Data and device
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing dataset CSV files (default: current directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    
    # Optuna parameters
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "grid", "random"],
        help="Sampler type (default: tpe)",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "noop", "percentile"],
        help="Pruner type (default: median)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Random seed for reproducibility (default: 46)",
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main tuning script."""
    args = parse_arguments()
    
    # Handle predefined config
    if args.config_type:
        config = get_search_config(args.config_type)
        n_trials = config["n_trials"]
        max_epochs = config["max_epochs"]
        if args.verbose:
            print(f"Using predefined config '{args.config_type}': {config['description']}\n")
    else:
        n_trials = args.n_trials
        max_epochs = args.max_epochs
    
    verbose = not args.no_verbose and args.verbose
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        architecture=args.architecture,
        n_trials=n_trials,
        max_epochs=max_epochs,
        data_dir=args.data_dir,
        device=args.device,
        verbose=verbose,
        seed=args.seed,
    )
    
    # Run optimization
    study = optimizer.run_optimization(
        sampler_type=args.sampler,
        pruner_type=args.pruner,
        n_jobs=args.n_jobs,
    )
    
    # Export results
    optimizer.export_best_config(study)


if __name__ == "__main__":
    main()
