"""
Chemical Density Surrogate Model Training Script
=================================================

Public API for training individual models on the chemical density dataset.
Supports MLP, CNN, Multi-Scale CNN, and LightGBM architectures.

Usage:
    # Train with defaults (MLP, hidden=[16,32,8], relu)
    python train_api.py
    
    # Train with custom architecture
    python train_api.py --architecture cnn
    
    # Train MLP with custom layers
    python train_api.py --hidden-dims 64 128 64 --activation silu
    
    # Train with tuned config
    python train_api.py --tuned-dir results_tuning/mlp_...

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training import SimpleTrainer
from core import TrainingConfig
from models import AVAILABLE_ARCHITECTURES, ACTIVATION_TYPES


# =============================================================================
# DEFAULT CONFIGURATION - Edit these values to change defaults
# =============================================================================

@dataclass
class DefaultConfig:
    """Default configuration for training.
    
    Edit these values to change the default behavior when running:
        python train_api.py
    """
    # Model architecture
    architecture: str = "mlp"
    
    # MLP-specific: layer structure (only used when architecture="mlp")
    hidden_dims: List[int] = field(default_factory=lambda: [16, 32, 8])
    activation: str = "relu"  # Options: relu, silu, leakyrelu, swiglu
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    dropout_rate: float = 0.0
    
    # Data
    data_path: str = "dataset.csv"
    
    # Output
    output_dir: Optional[str] = None  # Auto-generated if None
    
    # Other
    seed: int = 42
    device: str = "auto"


DEFAULT = DefaultConfig()

# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (override defaults)."""
    parser = argparse.ArgumentParser(
        description="Train a chemical density surrogate model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python train_api.py                                    # Use defaults
  python train_api.py --architecture cnn                 # Train CNN
  python train_api.py --hidden-dims 64 128 64            # Custom MLP layers
  python train_api.py --activation silu --epochs 200     # SiLU activation, 200 epochs
  python train_api.py --tuned-dir results_tuning/mlp_... # Use tuned config

Default MLP architecture: {DEFAULT.hidden_dims} with {DEFAULT.activation} activation
        """,
    )
    
    # Model architecture
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default=None,
        choices=AVAILABLE_ARCHITECTURES,
        help=f"Model architecture (default: {DEFAULT.architecture})",
    )
    
    # MLP-specific arguments
    parser.add_argument(
        "--hidden-dims", "-hd",
        type=int,
        nargs="+",
        default=None,
        help=f"Hidden layer dimensions for MLP (default: {DEFAULT.hidden_dims})",
    )
    
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=ACTIVATION_TYPES,
        help=f"Activation function for MLP (default: {DEFAULT.activation})",
    )
    
    # Data
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help=f"Path to dataset (default: {DEFAULT.data_path})",
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
        default=None,
        help=f"Number of training epochs (default: {DEFAULT.epochs})",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help=f"Batch size (default: {DEFAULT.batch_size})",
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=None,
        help=f"Learning rate (default: {DEFAULT.learning_rate})",
    )
    
    parser.add_argument(
        "--weight-decay", "-wd",
        type=float,
        default=None,
        help=f"Weight decay (default: {DEFAULT.weight_decay})",
    )
    
    parser.add_argument(
        "--dropout", "-do",
        type=float,
        default=None,
        help=f"Dropout rate (default: {DEFAULT.dropout_rate})",
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
        help="Suppress output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Merge args with defaults (args override defaults)
    architecture = args.architecture or DEFAULT.architecture
    hidden_dims = args.hidden_dims or DEFAULT.hidden_dims
    activation = args.activation or DEFAULT.activation
    data_path = args.data_path or DEFAULT.data_path
    epochs = args.epochs or DEFAULT.epochs
    batch_size = args.batch_size or DEFAULT.batch_size
    learning_rate = args.learning_rate or DEFAULT.learning_rate
    weight_decay = args.weight_decay or DEFAULT.weight_decay
    dropout_rate = args.dropout if args.dropout is not None else DEFAULT.dropout_rate
    seed = args.seed or DEFAULT.seed
    device = args.device or DEFAULT.device
    output_dir = args.output_dir or DEFAULT.output_dir
    
    # Build model config based on architecture
    model_config = {}
    if architecture == "mlp":
        model_config = {
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout_rate": dropout_rate,
        }
    
    # Create training config
    config = TrainingConfig(
        architecture=architecture,
        model_config=model_config,
        data_path=data_path,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        output_dir=output_dir,
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
