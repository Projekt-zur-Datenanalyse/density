"""
Chemical Density Surrogate Model Training Script
=================================================

Public API for training individual models on the chemical density dataset.
Supports MLP, CNN, Multi-Scale CNN, and LightGBM architectures.

Usage:
    # Train with defaults
    python train.py --architecture mlp
    
    # Train with tuned config
    python train.py --architecture mlp --tuned-dir tuning_results/mlp_...
    
    # Train with custom parameters
    python train.py --architecture cnn --epochs 100 --batch-size 64

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training import SimpleTrainer
from core import TrainingConfig
from models import AVAILABLE_ARCHITECTURES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a chemical density surrogate model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --architecture mlp
  python train.py --architecture cnn --epochs 100
  python train.py --architecture mlp --tuned-dir tuning_results/mlp_20250101_120000
        """,
    )
    
    # Required
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        required=True,
        choices=AVAILABLE_ARCHITECTURES,
        help="Model architecture to train",
    )
    
    # Data
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default="dataset.csv",
        help="Path to dataset (default: dataset.csv)",
    )
    
    # Tuned config
    parser.add_argument(
        "--tuned-dir", "-t",
        type=str,
        default=None,
        help="Path to tuning results directory for loading optimized config",
    )
    
    parser.add_argument(
        "--tuned-rank", "-r",
        type=int,
        default=1,
        help="Rank of tuned config to use (default: 1 = best)",
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
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
    
    parser.add_argument(
        "--weight-decay", "-wd",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
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
    
    # Create config
    config = TrainingConfig(
        architecture=args.architecture,
        data_path=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Create trainer
    trainer = SimpleTrainer(config, verbose=not args.quiet)
    
    # Load tuned config if specified
    if args.tuned_dir:
        trainer.load_tuned_config(args.tuned_dir, rank=args.tuned_rank)
    
    # Run training
    results = trainer.run()
    
    if not args.quiet:
        print(f"\n✓ Training complete!")
        print(f"  Test RMSE: {results['test_rmse_denormalized']:.2f} kg/m³")
        print(f"  Results: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
