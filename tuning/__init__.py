"""
Tuning package for hyperparameter optimization.

This package provides Optuna-based hyperparameter tuning:
- Search space definitions for all architectures
- Objective functions for optimization
- Results management and saving

Usage:
    from tuning import HyperparameterTuner, SearchSpace
"""

from .search_spaces import SearchSpace, SEARCH_CONFIGS
from .objective import create_objective
from .manager import HyperparameterTuner, TuningResults

__all__ = [
    "SearchSpace",
    "SEARCH_CONFIGS",
    "create_objective",
    "HyperparameterTuner",
    "TuningResults",
]
