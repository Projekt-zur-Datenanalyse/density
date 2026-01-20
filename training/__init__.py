"""
Training package for Chemical Density Surrogate.

This package provides training functionality:
- Simple single-model training
- Deep ensemble training with uncertainty quantification

Usage:
    from training import SimpleTrainer, EnsembleTrainer
"""

from .simple import SimpleTrainer
from .ensemble import EnsembleTrainer

__all__ = ["SimpleTrainer", "EnsembleTrainer"]
