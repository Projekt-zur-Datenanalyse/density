"""Main hyperparameter tuning script using Optuna.

This script runs hyperparameter optimization for any of the model architectures:
- mlp: Multi-Layer Perceptron
- cnn: Convolutional Neural Network
- cnn_multiscale: Multi-Scale CNN
- lightgbm: LightGBM

Now uses the unified optuna_manager for consistent behavior.

Usage:
    python tune.py mlp --n-trials 50 --max-epochs 100
    python tune.py cnn --config-type balanced
    python tune.py lightgbm --n-best-save 10
"""

import argparse
import sys
from optuna_manager import OptunaOptimizer, find_latest_results_dir, SEARCH_SPACE_CONFIGS, get_search_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning using Optuna (unified manager)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune.py mlp --n-trials 50
  python tune.py cnn --config-type balanced
  python tune.py lightgbm --n-trials 100 --sampler tpe
  python tune.py cnn_multiscale --n-best-save 10
        """,
    )
    
    # Positional architecture argument
    parser.add_argument(
        "architecture",
        type=str,
        choices=["mlp", "cnn", "cnn_multiscale", "lightgbm"],
        help="Model architecture to tune",
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
    parser.add_argument(
        "--n-best-save",
        type=int,
        default=5,
        help="Number of best configs to save (default: 5)",
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
        help="Master seed for reproducibility (default: 46)",
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
    else:
        n_trials = args.n_trials
        max_epochs = args.max_epochs
    
    # Create optimizer
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
    
    # Run optimization
    study, results = optimizer.run_optimization(
        sampler_type=args.sampler,
        pruner_type=args.pruner,
        n_jobs=args.n_jobs,
    )
    
    print(f"\n✓ Tuning completed successfully!")
    print(f"✓ Results saved to: {optimizer.results_dir}")


if __name__ == "__main__":
    main()
