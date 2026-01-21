#!/usr/bin/env python3
"""
Comparison Study API - Big MLP vs Small MLP Architecture Analysis.

This script runs comprehensive comparison experiments between:
1. Big MLP (4→512→2048→254→64→4→1, ~1.6M params)
2. Small MLP (4→16→32→8→1, ~900 params)

Under different conditions:
- With/without normalization
- Full dataset vs density-constrained (600-750 kg/m³)
- Different activation functions (relu, silu, leakyrelu)

Key factors being tested:
- Model size impact: 1.6M params vs 900 params
- Normalization importance
- Training data distribution effects (full vs constrained)
- Activation function performance

Usage:
    python comparison_api.py
    python comparison_api.py --num-epochs 200
    python comparison_api.py --quick  # Fast test run
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.comparison_study import ComparisonConfig, run_comparison_study


# ============================================================================
# CONFIGURATION - Edit these values to customize the comparison study
# ============================================================================

@dataclass
class DefaultConfig(ComparisonConfig):
    """Default configuration for comparison study.
    
    Edit these values directly to customize your experiments.
    """
    
    # Output
    output_dir: str = "results_comparison"
    
    # Dataset
    data_path: str = "dataset.csv"
    validation_split: float = 0.15
    test_split: float = 0.10
    batch_size: int = 64
    seed: int = 46
    
    # Training
    num_epochs: int = 300
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Which experiments to run
    run_size_comparison: bool = True          # Big vs Small under same conditions
    run_normalization_ablation: bool = True   # With vs Without normalization
    run_activation_comparison: bool = True    # Different activations
    run_data_constraint_study: bool = True    # Full vs Constrained (600-750)
    
    # Architectures
    big_hidden_dims: List[int] = field(default_factory=lambda: [512, 2048, 254, 64, 4])
    small_hidden_dims: List[int] = field(default_factory=lambda: [16, 32, 8])
    
    # Activations to compare
    activations: List[str] = field(default_factory=lambda: ["relu", "leakyrelu", "silu"])
    
    # Density constraint range
    density_min: float = 600.0
    density_max: float = 750.0
    
    # Device
    device: str = "auto"


def main():
    """Run comparison study with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comparison study between Big MLP and Small MLP architectures"
    )
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of training epochs (default: 300)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with 50 epochs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--skip-size", action="store_true",
                        help="Skip Big vs Small comparison")
    parser.add_argument("--skip-normalization", action="store_true",
                        help="Skip normalization ablation experiments")
    parser.add_argument("--skip-activation", action="store_true",
                        help="Skip activation function comparison")
    parser.add_argument("--skip-constrained", action="store_true",
                        help="Skip density-constrained experiments")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Create config from defaults
    config = DefaultConfig()
    
    # Apply command line overrides
    if args.quick:
        config.num_epochs = 50
        print("Quick mode: Using 50 epochs")
    elif args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    config.run_size_comparison = not args.skip_size
    config.run_normalization_ablation = not args.skip_normalization
    config.run_activation_comparison = not args.skip_activation
    config.run_data_constraint_study = not args.skip_constrained
    config.device = args.device
    
    # Print configuration summary
    print("="*80)
    print("COMPARISON STUDY: Big vs Small MLP")
    print("="*80)
    print(f"Output directory: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print()
    print("Experiments:")
    print(f"  - Size comparison (Big vs Small): {config.run_size_comparison}")
    print(f"  - Normalization ablation: {config.run_normalization_ablation}")
    print(f"  - Activation comparison: {config.run_activation_comparison}")
    print(f"  - Data constraint study: {config.run_data_constraint_study}")
    print()
    print("Architectures:")
    print(f"  - Big:   4 → {' → '.join(map(str, config.big_hidden_dims))} → 1 (~1.6M params)")
    print(f"  - Small: 4 → {' → '.join(map(str, config.small_hidden_dims))} → 1 (~900 params)")
    print()
    print("Activations:", ", ".join(config.activations))
    print("="*80)
    print()
    
    # Run comparison study
    results = run_comparison_study(config)
    
    print("\n" + "="*80)
    print("COMPARISON STUDY COMPLETE")
    print("="*80)
    print(f"Results saved to: {config.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
