"""
Chemical Density Deep Ensemble Training Script
===============================================

Public API for training deep ensembles for uncertainty quantification.
Supports multiple architectures and ensemble configurations.

Usage:
    # Train ensemble with defaults
    python ensemble_api.py
    
    # Train CNN ensemble
    python ensemble_api.py --architecture cnn
    
    # Multi-architecture ensemble
    python ensemble_api.py --architectures mlp cnn --n-models 3

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training import EnsembleTrainer
from core import TrainingConfig
from models import AVAILABLE_ARCHITECTURES, ACTIVATION_TYPES


# =============================================================================
# DEFAULT CONFIGURATION - Edit these values to change defaults
# =============================================================================

@dataclass
class DefaultEnsembleConfig:
    """Default configuration for ensemble training.
    
    Edit these values to change the default behavior when running:
        python ensemble_api.py
    """
    # Ensemble configuration
    architecture: str = "mlp"  # Single architecture for homogeneous ensemble
    n_models: int = 5          # Number of ensemble members
    
    # MLP-specific defaults
    hidden_dims: List[int] = field(default_factory=lambda: [16, 32, 8])
    activation: str = "relu"
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    
    # Data
    data_path: str = "dataset.csv"
    
    # Output
    output_dir: Optional[str] = None
    
    # Other
    seed: int = 42
    device: str = "auto"


DEFAULT = DefaultEnsembleConfig()

# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (override defaults)."""
    parser = argparse.ArgumentParser(
        description="Train a deep ensemble for chemical density prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python ensemble_api.py                                  # Use defaults (5 MLP models)
  python ensemble_api.py --architecture cnn --n-models 3  # 3 CNN models
  python ensemble_api.py --architectures mlp cnn          # Mixed ensemble
  python ensemble_api.py --hidden-dims 64 128 64          # Custom MLP layers

Default: {DEFAULT.n_models} {DEFAULT.architecture} models with {DEFAULT.hidden_dims} architecture
        """,
    )
    
    # Architecture (single or multiple)
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default=None,
        choices=AVAILABLE_ARCHITECTURES,
        help=f"Single architecture for homogeneous ensemble (default: {DEFAULT.architecture})",
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
        default=None,
        help=f"Number of ensemble members (default: {DEFAULT.n_models})",
    )
    
    # MLP-specific
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
        default=None,
        help=f"Number of training epochs per model (default: {DEFAULT.epochs})",
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
    
    # Determine architecture(s)
    if args.architectures:
        architectures = args.architectures
    elif args.architecture:
        architectures = [args.architecture]
    else:
        architectures = [DEFAULT.architecture]
    
    # Merge args with defaults
    n_models = args.n_models or DEFAULT.n_models
    hidden_dims = args.hidden_dims or DEFAULT.hidden_dims
    activation = args.activation or DEFAULT.activation
    data_path = args.data_path or DEFAULT.data_path
    epochs = args.epochs or DEFAULT.epochs
    batch_size = args.batch_size or DEFAULT.batch_size
    learning_rate = args.learning_rate or DEFAULT.learning_rate
    seed = args.seed or DEFAULT.seed
    device = args.device or DEFAULT.device
    output_dir = args.output_dir or DEFAULT.output_dir
    
    # Build model config for MLP
    model_config = {}
    if "mlp" in architectures:
        model_config = {
            "hidden_dims": hidden_dims,
            "activation": activation,
        }
    
    # Build training config
    training_config = TrainingConfig(
        data_path=data_path,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )
    
    # Create ensemble trainer
    trainer = EnsembleTrainer(
        architectures=architectures,
        n_models=n_models,
        model_config=model_config,
        training_config=training_config,
        tuned_config_dir=args.tuned_dir,
        output_dir=output_dir if output_dir else "results_ensemble",
        master_seed=seed,
    )
    
    # Train ensemble
    results = trainer.run()
    
    if not args.quiet:
        print(f"\n✓ Ensemble training complete!")
        print(f"  Ensemble RMSE: {results['ensemble_rmse_denormalized']:.2f} kg/m³")
        print(f"  Mean Uncertainty: {results.get('mean_std', 0):.2f} kg/m³")
        print(f"  Results: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
