"""
Chemical Density Hyperparameter Tuning Script
==============================================

Public API for hyperparameter optimization using Optuna.
Finds optimal configurations for any supported architecture.

Usage:
    # Quick tune with defaults
    python tune_api.py
    
    # Tune CNN
    python tune_api.py --architecture cnn
    
    # Production tuning
    python tune_api.py --trials 200 --epochs 100

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tuning import HyperparameterTuner, SEARCH_CONFIGS
from models import AVAILABLE_ARCHITECTURES


# =============================================================================
# DEFAULT CONFIGURATION - Edit these values to change defaults
# =============================================================================

@dataclass
class DefaultTuningConfig:
    """Default configuration for hyperparameter tuning.
    
    Edit these values to change the default behavior when running:
        python tune_api.py
    """
    # Architecture to tune
    architecture: str = "mlp"
    
    # Tuning parameters
    n_trials: int = 50
    max_epochs: int = 100
    n_best_save: int = 5
    
    # Sampler/Pruner
    sampler: str = "tpe"   # "tpe" or "random"
    pruner: str = "median"  # "median", "noop", or "percentile"
    
    # Data
    data_path: str = "dataset.csv"
    
    # Output
    output_dir: Optional[str] = None
    
    # Other
    seed: int = 46
    device: str = "auto"


DEFAULT = DefaultTuningConfig()

# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (override defaults)."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for chemical density models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python tune_api.py                          # Use defaults ({DEFAULT.n_trials} trials)
  python tune_api.py --architecture cnn       # Tune CNN
  python tune_api.py --preset balanced        # Use preset config
  python tune_api.py --trials 100 --epochs 50 # Custom search

Presets:
  minimal:    10 trials, 20 epochs (quick testing)
  balanced:   50 trials, 50 epochs (standard tuning)
  extensive:  100 trials, 100 epochs (comprehensive)
  production: 200 trials, 150 epochs (full search)

Default: {DEFAULT.architecture} with {DEFAULT.n_trials} trials, {DEFAULT.max_epochs} epochs
        """,
    )
    
    # Architecture
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default=None,
        choices=AVAILABLE_ARCHITECTURES,
        help=f"Model architecture to tune (default: {DEFAULT.architecture})",
    )
    
    # Tuning parameters
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=None,
        help=f"Number of optimization trials (default: {DEFAULT.n_trials})",
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help=f"Max epochs per trial (default: {DEFAULT.max_epochs})",
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
        default=None,
        help=f"Number of best configs to save (default: {DEFAULT.n_best_save})",
    )
    
    # Data
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help=f"Path to dataset (default: {DEFAULT.data_path})",
    )
    
    # Sampler and pruner
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["tpe", "random"],
        help=f"Optuna sampler (default: {DEFAULT.sampler})",
    )
    
    parser.add_argument(
        "--pruner",
        type=str,
        default=None,
        choices=["median", "noop", "percentile"],
        help=f"Optuna pruner (default: {DEFAULT.pruner})",
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
        default=None,
        help=f"Random seed (default: {DEFAULT.seed})",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu", "mps"],
        help=f"Device to use (default: {DEFAULT.device})",
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
    
    # Merge args with defaults
    architecture = args.architecture or DEFAULT.architecture
    data_path = args.data_path or DEFAULT.data_path
    n_best = args.n_best or DEFAULT.n_best_save
    sampler = args.sampler or DEFAULT.sampler
    pruner = args.pruner or DEFAULT.pruner
    seed = args.seed or DEFAULT.seed
    device = args.device or DEFAULT.device
    output_dir = args.output_dir or DEFAULT.output_dir
    
    # Determine trials and epochs (preset overrides defaults, args override preset)
    if args.preset:
        preset = SEARCH_CONFIGS[args.preset]
        n_trials = args.trials if args.trials is not None else preset["n_trials"]
        max_epochs = args.epochs if args.epochs is not None else preset["max_epochs"]
        if not args.quiet:
            print(f"Using preset '{args.preset}': {preset['description']}")
    else:
        n_trials = args.trials if args.trials is not None else DEFAULT.n_trials
        max_epochs = args.epochs if args.epochs is not None else DEFAULT.max_epochs
    
    # Create tuner
    tuner = HyperparameterTuner(
        architecture=architecture,
        n_trials=n_trials,
        max_epochs=max_epochs,
        n_best_save=n_best,
        data_path=data_path,
        device=device,
        seed=seed,
        output_dir=output_dir,
    )
    
    # Run optimization
    study, metadata = tuner.run(
        sampler=sampler,
        pruner=pruner,
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
