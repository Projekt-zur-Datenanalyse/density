"""Run training and display comprehensive RMSE analysis."""

import sys
import json
from pathlib import Path
import numpy as np
from analyze_data_ranges import analyze_data_ranges


def display_rmse_interpretation(rmse_denorm: float, data_info: dict):
    """Display RMSE in perspective of the data.
    
    Args:
        rmse_denorm: Denormalized RMSE value in kg/m³
        data_info: Dictionary with data statistics from analyze_data_ranges
    """
    density_range = data_info['density']['range']
    density_std = data_info['density']['std']
    density_mean = data_info['density']['mean']
    
    print("\n" + "=" * 80)
    print("RMSE PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    print(f"\nTest RMSE: {rmse_denorm:.2f} kg/m³")
    print(f"\nPerspective:")
    print(f"  • {(rmse_denorm/density_range)*100:.1f}% of density range ({density_range:.2f} kg/m³)")
    print(f"  • {(rmse_denorm/density_std)*100:.1f}% of std deviation ({density_std:.2f} kg/m³)")
    print(f"  • {(rmse_denorm/density_mean)*100:.1f}% of mean density ({density_mean:.2f} kg/m³)")
    
    # Comparison to baseline
    naive_rmse = density_std
    improvement = ((naive_rmse - rmse_denorm) / naive_rmse) * 100
    print(f"  • {improvement:.1f}% better than naive baseline ({naive_rmse:.2f} kg/m³)")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if rmse_denorm < 0.5 * density_std:
        quality = "EXCELLENT 🌟"
    elif rmse_denorm < 1.0 * density_std:
        quality = "VERY GOOD ⭐"
    elif rmse_denorm < 1.5 * density_std:
        quality = "GOOD ✓"
    else:
        quality = "ACCEPTABLE"
    print(f"  • {quality}")
    print(f"  • RMSE = {(rmse_denorm/density_std):.2f}x Std Dev")
    
    # Practical examples
    print(f"\nPractical Examples:")
    example_densities = [
        ("Low (25%)", density_mean * 0.25),
        ("Low-Mid (50%)", density_mean * 0.50),
        ("Mean", density_mean),
        ("Mid-High (75%)", density_mean * 0.75),
        ("High (90%)", density_mean * 0.90),
    ]
    
    print(f"\n{'Density Level':<20} {'True Density':<15} {'Prediction Range':<30} {'Relative Error':<15}")
    print("-" * 80)
    for label, true_val in example_densities:
        lower = max(0, true_val - rmse_denorm)
        upper = true_val + rmse_denorm
        rel_error = (rmse_denorm / true_val * 100) if true_val > 0 else 0
        pred_range = f"[{lower:.1f} - {upper:.1f}]"
        print(f"{label:<20} {true_val:<15.1f} {pred_range:<30} ±{rel_error:.1f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main analysis workflow."""
    print("\nAnalyzing data ranges...")
    data_info = analyze_data_ranges()
    
    # Check for recent results
    results_dir = Path("./results")
    if results_dir.exists():
        test_results_file = results_dir / "test_results.json"
        if test_results_file.exists():
            with open(test_results_file, 'r') as f:
                results = json.load(f)
                test_rmse_norm = results.get('test_rmse')
                
                # Denormalize RMSE
                target_std = data_info['density']['std']
                test_rmse_denorm = test_rmse_norm * target_std
                
                print(f"\nLoaded test results from {test_results_file}")
                print(f"Normalized RMSE: {test_rmse_norm:.6f}")
                print(f"Denormalized RMSE: {test_rmse_denorm:.2f} kg/m³")
                
                display_rmse_interpretation(test_rmse_denorm, data_info)


if __name__ == "__main__":
    main()
