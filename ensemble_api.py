"""
Chemical Density Deep Ensemble Training Script
===============================================

Public API for training deep ensembles for uncertainty quantification.
Supports multiple architectures and ensemble configurations.

Usage:
    # Train ensemble with defaults
    python ensemble.py --architecture mlp --n-models 5
    
    # Train ensemble with tuned configs
    python ensemble.py --architecture mlp --tuned-dir tuning_results/mlp_...
    
    # Multi-architecture ensemble
    python ensemble.py --architectures mlp cnn --n-models 3

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training import EnsembleTrainer
from models import AVAILABLE_ARCHITECTURES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a deep ensemble for chemical density prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ensemble.py --architecture mlp --n-models 5
  python ensemble.py --architecture cnn --tuned-dir tuning_results/cnn_...
  python ensemble.py --architectures mlp cnn cnn_multiscale --n-models 2
        """,
    )
    
    # Architecture (single or multiple)
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default=None,
        choices=AVAILABLE_ARCHITECTURES,
        help="Single architecture for homogeneous ensemble",
    )
    
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        default=None,
        choices=AVAILABLE_ARCHITECTURES,
        help="Multiple architectures for heterogeneous ensemble",
    )
    
    # Ensemble configuration
    parser.add_argument(
        "--n-models", "-n",
        type=int,
        default=5,
        help="Number of ensemble members (default: 5)",
    )
    
    # Data
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default="dataset.csv",
        help="Path to dataset (default: dataset.csv)",
    )
    
    # Tuned configs
    parser.add_argument(
        "--tuned-dir", "-t",
        type=str,
        default=None,
        help="Path to tuning results for loading configs",
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs per model (default: 100)",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
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
        default=42,
        help="Random seed (default: 42)",
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
        help="Suppress output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine architecture(s)
    if args.architectures:
        architectures = args.architectures
    elif args.architecture:
        architectures = [args.architecture]
    else:
        print("Error: Must specify --architecture or --architectures")
        sys.exit(1)
    
    # Create ensemble trainer
    trainer = EnsembleTrainer(
        architectures=architectures,
        n_models=args.n_models,
        data_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    # Load tuned configs if specified
    if args.tuned_dir:
        trainer.load_tuned_configs(args.tuned_dir)
    
    # Train ensemble
    results = trainer.train()
    
    if not args.quiet:
        print(f"\n✓ Ensemble training complete!")
        print(f"  Ensemble RMSE: {results['ensemble_rmse_denormalized']:.2f} kg/m³")
        print(f"  Mean Uncertainty: {results.get('mean_std', 0):.2f} kg/m³")
        print(f"  Results: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
