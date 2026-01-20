"""
Chemical Density Hyperparameter Tuning Script
==============================================

Public API for hyperparameter optimization using Optuna.
Finds optimal configurations for any supported architecture.

Usage:
    # Quick tune with defaults
    python tune.py --architecture mlp --trials 50
    
    # Production tuning
    python tune.py --architecture cnn --trials 200 --epochs 100

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tuning import HyperparameterTuner, SEARCH_CONFIGS
from models import AVAILABLE_ARCHITECTURES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for chemical density models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune.py --architecture mlp --trials 50
  python tune.py --architecture cnn --preset balanced
  python tune.py --architecture lightgbm --trials 100 --epochs 50

Presets:
  minimal:    10 trials, 20 epochs (quick testing)
  balanced:   50 trials, 50 epochs (standard tuning)
  extensive:  100 trials, 100 epochs (comprehensive)
  production: 200 trials, 150 epochs (full search)
        """,
    )
    
    # Required
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        required=True,
        choices=AVAILABLE_ARCHITECTURES,
        help="Model architecture to tune",
    )
    
    # Tuning parameters
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=None,
        help="Number of optimization trials",
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Max epochs per trial",
    )
    
    parser.add_argument(
        "--preset", "-p",
        type=str,
        default=None,
        choices=list(SEARCH_CONFIGS.keys()),
        help="Use a preset configuration",
    )
    
    parser.add_argument(
        "--n-best", "-b",
        type=int,
        default=5,
        help="Number of best configs to save (default: 5)",
    )
    
    # Data
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default="dataset.csv",
        help="Path to dataset (default: dataset.csv)",
    )
    
    # Sampler and pruner
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler (default: tpe)",
    )
    
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "noop", "percentile"],
        help="Optuna pruner (default: median)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    
    # Other
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=46,
        help="Random seed (default: 46)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use (default: auto)",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine trials and epochs
    if args.preset:
        preset = SEARCH_CONFIGS[args.preset]
        n_trials = args.trials or preset["n_trials"]
        max_epochs = args.epochs or preset["max_epochs"]
        if not args.quiet:
            print(f"Using preset '{args.preset}': {preset['description']}")
    else:
        n_trials = args.trials or 50
        max_epochs = args.epochs or 100
    
    # Create tuner
    tuner = HyperparameterTuner(
        architecture=args.architecture,
        n_trials=n_trials,
        max_epochs=max_epochs,
        n_best_save=args.n_best,
        data_path=args.data_path,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run optimization
    study, metadata = tuner.run(
        sampler=args.sampler,
        pruner=args.pruner,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print(f"\n✓ Tuning complete!")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best val RMSE: {study.best_value:.6f}")
        print(f"  Saved {len(metadata['configs'])} best configs")
        print(f"  Results: {tuner.output_dir}")
    
    return study, metadata


if __name__ == "__main__":
    main()
