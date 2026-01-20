"""
Chemical Density Analysis Script
================================

Public API for analyzing data, models, and results.
Provides utilities for data exploration and model evaluation.

Usage:
    # Analyze dataset
    python analyze.py data --path dataset.csv
    
    # Analyze model results
    python analyze.py model --results-dir results_mlp/
    
    # Compare models
    python analyze.py compare --dirs results_mlp results_cnn results_lightgbm

Author: Chemical Density Surrogate Project
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import DataAnalyzer, ModelAnalyzer, Plotter
from analysis.model_analysis import compare_models, analyze_ensemble_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analysis tools for chemical density surrogate models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py data --path dataset.csv --report
  python analyze.py model --results-dir results_mlp/
  python analyze.py compare --dirs results_mlp results_cnn
  python analyze.py ensemble --results-dir ensemble_results/
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Analysis command")
    
    # Data analysis
    data_parser = subparsers.add_parser("data", help="Analyze dataset")
    data_parser.add_argument(
        "--path", "-p",
        type=str,
        default="dataset.csv",
        help="Path to dataset",
    )
    data_parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate detailed report",
    )
    data_parser.add_argument(
        "--output", "-o",
        type=str,
        default="analysis/data_report.txt",
        help="Report output path",
    )
    
    # Model analysis
    model_parser = subparsers.add_parser("model", help="Analyze trained model")
    model_parser.add_argument(
        "--results-dir", "-d",
        type=str,
        required=True,
        help="Path to results directory",
    )
    model_parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate plots",
    )
    
    # Model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="Result directories to compare",
    )
    compare_parser.add_argument(
        "--metric", "-m",
        type=str,
        default="rmse_denormalized",
        help="Metric to compare (default: rmse_denormalized)",
    )
    compare_parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate comparison plot",
    )
    
    # Ensemble analysis
    ensemble_parser = subparsers.add_parser("ensemble", help="Analyze ensemble")
    ensemble_parser.add_argument(
        "--results-dir", "-d",
        type=str,
        required=True,
        help="Path to ensemble results directory",
    )
    
    return parser.parse_args()


def analyze_data(args):
    """Analyze dataset."""
    analyzer = DataAnalyzer(args.path)
    analyzer.load_data()
    
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS")
    print(f"{'='*60}\n")
    
    # Summary
    summary = analyzer.get_summary()
    print(f"Path:     {args.path}")
    print(f"Samples:  {summary['n_samples']}")
    print(f"Features: {summary['features']}")
    print(f"Target:   {summary['target']}\n")
    
    # Ranges
    print("Feature Ranges:")
    ranges = analyzer.get_feature_ranges()
    for col, stats in ranges.items():
        print(f"  {col}: [{stats['min']:.4f}, {stats['max']:.4f}] "
              f"(mean={stats['mean']:.4f}, std={stats['std']:.4f})")
    
    # Correlations
    print("\nTarget Correlations:")
    corrs = analyzer.get_target_correlations()
    for feat, corr in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat}: {corr:+.4f}")
    
    # Generate report if requested
    if args.report:
        report_path = analyzer.save_report(args.output)
        print(f"\nReport saved to: {report_path}")
    
    print(f"\n{'='*60}\n")


def analyze_model(args):
    """Analyze trained model."""
    analyzer = ModelAnalyzer(args.results_dir)
    
    print(f"\n{'='*60}")
    print("MODEL ANALYSIS")
    print(f"{'='*60}\n")
    
    # Test results
    test_results = analyzer.load_test_results()
    print(f"Results Directory: {args.results_dir}\n")
    print("Test Metrics:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Model config
    config = analyzer.load_model_config()
    if config:
        print(f"\nModel Config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Residuals
    residuals = analyzer.compute_residuals()
    if residuals:
        print(f"\nResidual Statistics:")
        print(f"  Mean:   {residuals['mean']:.4f}")
        print(f"  Std:    {residuals['std']:.4f}")
        print(f"  MAE:    {residuals['mae']:.4f}")
        print(f"  RMSE:   {residuals['rmse']:.4f}")
    
    # Generate plots if requested
    if args.plot:
        plotter = Plotter(output_dir=f"{args.results_dir}/plots")
        
        history = analyzer.load_training_history()
        if history:
            plotter.training_history(history)
        
        predictions = analyzer.load_predictions()
        if predictions is not None:
            preds = predictions["predictions"].numpy()
            targets = predictions["targets"].numpy()
            plotter.predictions_vs_actual(preds, targets)
            plotter.residual_plot(preds, targets)
        
        print(f"\nPlots saved to: {args.results_dir}/plots")
    
    print(f"\n{'='*60}\n")


def compare_models_cmd(args):
    """Compare multiple models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    results = compare_models(args.dirs, metric=args.metric)
    
    print(f"Metric: {args.metric}\n")
    print("Rankings:")
    for i, r in enumerate(results["rankings"], 1):
        print(f"  {i}. {r['architecture']}: {r['metric_value']:.4f} ({r['directory']})")
    
    if results["errors"]:
        print("\nErrors:")
        for e in results["errors"]:
            print(f"  {e['directory']}: {e['error']}")
    
    if results["best"]:
        print(f"\nBest: {results['best']['architecture']} "
              f"({results['best']['metric_value']:.4f})")
    
    # Generate plot if requested
    if args.plot:
        comparison_data = {
            r['architecture']: r['metric_value']
            for r in results["rankings"]
        }
        plotter = Plotter(output_dir="analysis/plots")
        plot_path = plotter.model_comparison(comparison_data)
        print(f"\nPlot saved to: {plot_path}")
    
    print(f"\n{'='*60}\n")


def analyze_ensemble_cmd(args):
    """Analyze ensemble results."""
    print(f"\n{'='*60}")
    print("ENSEMBLE ANALYSIS")
    print(f"{'='*60}\n")
    
    results = analyze_ensemble_results(args.results_dir)
    
    print(f"Ensemble Directory: {args.results_dir}")
    print(f"Number of Members: {results['n_members']}\n")
    
    print("Ensemble Metrics:")
    for key, value in results["ensemble_metrics"].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    if results["individual_stats"]:
        print("\nIndividual Model Stats:")
        stats = results["individual_stats"]
        print(f"  Mean RMSE: {stats['mean_rmse']:.4f}")
        print(f"  Std RMSE:  {stats['std_rmse']:.4f}")
        print(f"  Min RMSE:  {stats['min_rmse']:.4f}")
        print(f"  Max RMSE:  {stats['max_rmse']:.4f}")
    
    print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command is None:
        print("Error: Please specify a command (data, model, compare, ensemble)")
        print("Use --help for more information")
        sys.exit(1)
    
    if args.command == "data":
        analyze_data(args)
    elif args.command == "model":
        analyze_model(args)
    elif args.command == "compare":
        compare_models_cmd(args)
    elif args.command == "ensemble":
        analyze_ensemble_cmd(args)


if __name__ == "__main__":
    main()
