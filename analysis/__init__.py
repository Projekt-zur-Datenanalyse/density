"""
Analysis package for data exploration and model evaluation.

This package provides tools for:
- Data analysis and exploration
- Model performance analysis
- Visualization and plotting

Usage:
    from analysis import DataAnalyzer, ModelAnalyzer, Plotter
"""

from .data_analysis import DataAnalyzer, analyze_data_ranges
from .model_analysis import ModelAnalyzer
from .plotting import Plotter, create_comparison_plot, create_residual_plot

__all__ = [
    "DataAnalyzer",
    "ModelAnalyzer",
    "Plotter",
    "analyze_data_ranges",
    "create_comparison_plot",
    "create_residual_plot",
]
