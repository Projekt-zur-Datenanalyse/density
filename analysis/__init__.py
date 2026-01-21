"""
Analysis package for data exploration and model evaluation.

This package provides tools for:
- Data analysis and exploration
- Model performance analysis
- Visualization and plotting
- Comparison studies between different approaches

Usage:
    from analysis import DataAnalyzer, ModelAnalyzer, Plotter
    from analysis import ComparisonStudy, ComparisonConfig, run_comparison_study
"""

from .data_analysis import DataAnalyzer, analyze_data_ranges
from .model_analysis import ModelAnalyzer
from .plotting import Plotter, create_comparison_plot, create_residual_plot
from .comparison_study import ComparisonStudy, ComparisonConfig, run_comparison_study

__all__ = [
    "DataAnalyzer",
    "ModelAnalyzer",
    "Plotter",
    "analyze_data_ranges",
    "create_comparison_plot",
    "create_residual_plot",
    "ComparisonStudy",
    "ComparisonConfig",
    "run_comparison_study",
]
