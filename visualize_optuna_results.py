"""Visualize Optuna hyperparameter tuning results for all architectures."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import sqlite3


def load_optuna_results():
    """Load the latest Optuna results for each architecture.
    
    Returns:
        Dictionary mapping architecture names to their tuning results
    """
    architectures = ['mlp', 'cnn', 'cnn_multiscale', 'lightgbm']
    results = {}
    
    for arch in architectures:
        # Find latest optuna results directory
        if arch == "cnn":
            # Special handling for cnn vs cnn_multiscale
            optuna_dirs = [p for p in Path(".").glob(f"optuna_results_cnn_*") if p.is_dir()]
            optuna_dirs = [p for p in optuna_dirs if "_multiscale_" not in p.name]
            optuna_dirs = sorted(optuna_dirs, key=lambda p: p.name, reverse=True)
        elif arch == "cnn_multiscale":
            optuna_dirs = [p for p in Path(".").glob(f"optuna_results_cnn_multiscale_*") if p.is_dir()]
            optuna_dirs = sorted(optuna_dirs, key=lambda p: p.name, reverse=True)
        else:
            optuna_dirs = [p for p in Path(".").glob(f"optuna_results_{arch}_*") if p.is_dir()]
            optuna_dirs = sorted(optuna_dirs, key=lambda p: p.name, reverse=True)
        
        if not optuna_dirs:
            continue
        
        latest_dir = optuna_dirs[0]
        summary_file = latest_dir / "optuna_summary.json"
        best_config_file = latest_dir / "best_config.json"
        
        if not summary_file.exists() or not best_config_file.exists():
            continue
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            with open(best_config_file, 'r') as f:
                best_config = json.load(f)
            
            # Try to load trial history from database
            db_file = latest_dir / "study.db"
            trials_data = []
            if db_file.exists():
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    # First, let's see what columns are available
                    cursor.execute("PRAGMA table_info(trials)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if 'value' in columns:
                        cursor.execute("""
                            SELECT value FROM trials 
                            WHERE state = 'COMPLETE' 
                            ORDER BY number ASC
                        """)
                        trials_data = [row[0] for row in cursor.fetchall() if row[0] is not None]
                    
                    conn.close()
                except Exception as e:
                    # Silently skip if database access fails - will just use empty trials_data
                    pass
            
            results[arch] = {
                'directory': str(latest_dir),
                'summary': summary,
                'best_config': best_config,
                'trials_data': trials_data,
            }
        except Exception as e:
            print(f"Warning: Could not load results for {arch}: {e}")
    
    return results


def create_comprehensive_plot(results: dict, output_file: str = "optuna_results_visualization.png"):
    """Create comprehensive visualization of Optuna tuning results.
    
    Args:
        results: Dictionary of tuning results from load_optuna_results()
        output_file: Output filename for the plot
    """
    if not results:
        print("No Optuna results found. Run tuning first with: python tune.py")
        return
    
    architectures = sorted(results.keys())
    n_archs = len(architectures)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Color scheme for architectures
    colors = {
        'mlp': '#2ecc71',           # Green
        'cnn': '#3498db',           # Blue
        'cnn_multiscale': '#f39c12', # Orange
        'lightgbm': '#e74c3c',      # Red
    }
    
    # ============ TOP ROW: Performance Metrics ============
    
    # 1. Best Test RMSE Comparison
    ax1 = plt.subplot(2, 3, 1)
    test_rmses = [
        results[arch]['best_config']['metrics']['test_rmse_denorm']
        for arch in architectures
    ]
    bars1 = ax1.bar(architectures, test_rmses, color=[colors[a] for a in architectures], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
    ax1.set_title('Best Test RMSE After Tuning', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, rmse in zip(bars1, test_rmses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Best Trial Number
    ax2 = plt.subplot(2, 3, 2)
    trial_numbers = [
        results[arch]['summary']['best_trial']['number']
        for arch in architectures
    ]
    bars2 = ax2.bar(architectures, trial_numbers, color=[colors[a] for a in architectures], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Trial Number', fontsize=11, fontweight='bold')
    ax2.set_title('Best Trial Number (out of N trials)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels and total trials
    for i, (bar, trial_num) in enumerate(zip(bars2, trial_numbers)):
        height = bar.get_height()
        n_trials = results[architectures[i]]['summary']['n_trials']
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'#{int(trial_num)}\n({n_trials} total)', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Validation vs Test RMSE
    ax3 = plt.subplot(2, 3, 3)
    val_rmses = [
        results[arch]['summary']['best_trial']['value']  # This is the normalized validation RMSE
        for arch in architectures
    ]
    target_stds = [
        results[arch]['summary'].get('target_std', 1.0)
        for arch in architectures
    ]
    val_rmses_denorm = [v * std for v, std in zip(val_rmses, target_stds)]
    
    x = np.arange(len(architectures))
    width = 0.35
    bars_val = ax3.bar(x - width/2, val_rmses_denorm, width, label='Val RMSE', 
                       color=[colors[a] for a in architectures], alpha=0.7, edgecolor='black', linewidth=1)
    bars_test = ax3.bar(x + width/2, test_rmses, width, label='Test RMSE',
                        color=[colors[a] for a in architectures], alpha=0.95, edgecolor='black', linewidth=1)
    
    ax3.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
    ax3.set_title('Validation vs Test RMSE', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(architectures)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ============ BOTTOM ROW: Convergence and Summary ============
    
    # 4. Optimization Convergence (Best Value Over Trials)
    ax4 = plt.subplot(2, 3, 4)
    
    has_trials = False
    for arch in architectures:
        trials_data = results[arch].get('trials_data', [])
        if trials_data:
            has_trials = True
            # Convert to denormalized RMSE for better readability
            target_std = results[arch]['summary'].get('target_std', 1.0)
            trials_denorm = [v * target_std for v in trials_data]
            
            # Plot cumulative minimum (best value so far)
            best_so_far = np.minimum.accumulate(trials_denorm)
            ax4.plot(best_so_far, marker='o', markersize=2, label=arch.upper(), 
                    color=colors[arch], linewidth=2.5, alpha=0.8)
    
    if has_trials:
        ax4.legend(fontsize=9, loc='best')
    else:
        ax4.text(0.5, 0.5, 'Trial history not available\n(database or data inaccessible)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
    
    ax4.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Best RMSE So Far (kg/m³)', fontsize=11, fontweight='bold')
    ax4.set_title('Optimization Convergence', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    
    # 5. Number of Trials
    ax5 = plt.subplot(2, 3, 5)
    n_trials_list = [
        results[arch]['summary']['n_trials']
        for arch in architectures
    ]
    bars5 = ax5.bar(architectures, n_trials_list, color=[colors[a] for a in architectures], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
    ax5.set_title('Tuning Effort (Trials)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, n_trials in zip(bars5, n_trials_list):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(n_trials)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = "TUNING SUMMARY\n" + "="*60 + "\n\n"
    
    for arch in architectures:
        summary = results[arch]['summary']
        best_config = results[arch]['best_config']
        
        best_trial_num = summary['best_trial']['number']
        n_trials = summary['n_trials']
        test_rmse = best_config['metrics']['test_rmse_denorm']
        target_std = summary.get('target_std', 1.0)
        val_rmse_norm = summary['best_trial']['value']
        val_rmse_denorm = val_rmse_norm * target_std
        improvement = ((target_std - test_rmse) / target_std) * 100
        
        summary_text += f"{arch.upper()}\n"
        summary_text += f"  Trials: {best_trial_num + 1}/{n_trials}\n"
        summary_text += f"  Val RMSE: {val_rmse_denorm:.2f} kg/m³\n"
        summary_text += f"  Test RMSE: {test_rmse:.2f} kg/m³\n"
        summary_text += f"  Improvement: {improvement:.1f}%\n"
        summary_text += "\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to {output_file}")
    plt.close()


def create_detailed_architecture_plots(results: dict, output_dir: str = "."):
    """Create detailed plots for each architecture showing hyperparameters.
    
    Args:
        results: Dictionary of tuning results
        output_dir: Directory to save individual architecture plots
    """
    for arch in sorted(results.keys()):
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f'Optuna Tuning Results: {arch.upper()}', fontsize=14, fontweight='bold', y=0.98)
        
        summary = results[arch]['summary']
        best_config = results[arch]['best_config']
        
        # Top left: Key metrics
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('off')
        
        metrics_text = "KEY METRICS\n" + "-"*40 + "\n"
        metrics_text += f"Best Trial: #{summary['best_trial']['number']}\n"
        metrics_text += f"Total Trials: {summary['n_trials']}\n"
        metrics_text += f"Completed: {summary['completed_trials']}\n\n"
        metrics_text += "PERFORMANCE\n" + "-"*40 + "\n"
        val_rmse_norm = summary['best_trial']['value']
        target_std = summary.get('target_std', 1.0)
        val_rmse_denorm = val_rmse_norm * target_std
        metrics_text += f"Val RMSE (norm): {val_rmse_norm:.6f}\n"
        metrics_text += f"Val RMSE (denorm): {val_rmse_denorm:.2f} kg/m³\n"
        metrics_text += f"Test RMSE (norm): {best_config['metrics']['test_rmse']:.6f}\n"
        metrics_text += f"Test RMSE (denorm): {best_config['metrics']['test_rmse_denorm']:.2f} kg/m³\n"
        metrics_text += f"Test MAE: {best_config['metrics']['test_mae']:.2f}\n"
        
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=1))
        
        # Top right: Hyperparameters
        ax2 = plt.subplot(2, 2, 2)
        ax2.axis('off')
        
        hyper_text = "BEST HYPERPARAMETERS\n" + "-"*40 + "\n"
        for key, value in sorted(best_config['hyperparameters'].items()):
            if isinstance(value, float):
                if value < 0.01:
                    hyper_text += f"{key}: {value:.2e}\n"
                else:
                    hyper_text += f"{key}: {value:.4f}\n"
            else:
                hyper_text += f"{key}: {value}\n"
        
        ax2.text(0.05, 0.95, hyper_text, transform=ax2.transAxes,
                fontsize=9.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='black', linewidth=1))
        
        # Bottom left: Trial convergence
        ax3 = plt.subplot(2, 2, 3)
        trials_data = results[arch].get('trials_data', [])
        
        if trials_data:
            target_std = summary.get('target_std', 1.0)
            trials_denorm = [v * target_std for v in trials_data]
            best_so_far = np.minimum.accumulate(trials_denorm)
            
            ax3.plot(best_so_far, marker='o', markersize=3, label='Best RMSE', 
                    color='#3498db', linewidth=2.5)
            ax3.scatter(range(len(trials_denorm)), trials_denorm, alpha=0.3, s=20, label='Trial RMSE', color='#95a5a6')
            
            ax3.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
            ax3.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
            ax3.set_title('Convergence During Tuning', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(alpha=0.3, linestyle='--')
        else:
            ax3.text(0.5, 0.5, 'Trial history not available\n(database not accessible)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        
        # Bottom right: Statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        if trials_data:
            target_std = summary.get('target_std', 1.0)
            trials_denorm = [v * target_std for v in trials_data]
            
            stats_text = "TRIAL STATISTICS\n" + "-"*40 + "\n"
            stats_text += f"Best RMSE: {min(trials_denorm):.2f} kg/m³\n"
            stats_text += f"Worst RMSE: {max(trials_denorm):.2f} kg/m³\n"
            stats_text += f"Mean RMSE: {np.mean(trials_denorm):.2f} kg/m³\n"
            stats_text += f"Std Dev: {np.std(trials_denorm):.2f} kg/m³\n"
            stats_text += f"Median RMSE: {np.median(trials_denorm):.2f} kg/m³\n\n"
            stats_text += "IMPROVEMENT\n" + "-"*40 + "\n"
            improvement = ((target_std - min(trials_denorm)) / target_std) * 100
            stats_text += f"Over baseline (std): {improvement:.1f}%\n"
            degradation = ((max(trials_denorm) - min(trials_denorm)) / min(trials_denorm)) * 100
            stats_text += f"Worst vs Best: {degradation:.1f}%\n"
        else:
            stats_text = "Trial statistics\nnot available"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3, edgecolor='black', linewidth=1))
        
        plt.tight_layout()
        output_file = Path(output_dir) / f"optuna_results_{arch}_details.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Detailed plot for {arch} saved to {output_file}")
        plt.close()


def print_summary(results: dict):
    """Print text summary of tuning results.
    
    Args:
        results: Dictionary of tuning results
    """
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)
    
    architectures = sorted(results.keys())
    
    # Sort by test RMSE (best first)
    sorted_archs = sorted(
        architectures,
        key=lambda a: results[a]['best_config']['metrics']['test_rmse_denorm']
    )
    
    print(f"\n{'Rank':<6} {'Architecture':<20} {'Best Val RMSE':<18} {'Test RMSE':<15} {'Trial':<15}")
    print("-" * 85)
    
    for rank, arch in enumerate(sorted_archs, 1):
        best_trial_num = results[arch]['summary']['best_trial']['number']
        n_trials = results[arch]['summary']['n_trials']
        val_rmse_norm = results[arch]['summary']['best_trial']['value']
        target_std = results[arch]['summary'].get('target_std', 1.0)
        val_rmse_denorm = val_rmse_norm * target_std
        test_rmse = results[arch]['best_config']['metrics']['test_rmse_denorm']
        
        print(f"{rank:<6} {arch.upper():<20} {val_rmse_denorm:<17.2f} kg/m³ {test_rmse:<14.2f} kg/m³ "
              f"#{best_trial_num}/{n_trials}")
    
    print("\n" + "="*80)
    print("BEST ARCHITECTURE SUMMARY")
    print("="*80)
    
    best_arch = sorted_archs[0]
    best_summary = results[best_arch]['summary']
    best_config = results[best_arch]['best_config']
    
    print(f"\nWinner: {best_arch.upper()}")
    print(f"  Best Trial: #{best_summary['best_trial']['number']} out of {best_summary['n_trials']}")
    print(f"  Test RMSE: {best_config['metrics']['test_rmse_denorm']:.2f} kg/m³")
    print(f"  Best Hyperparameters:")
    for key, value in sorted(best_config['hyperparameters'].items()):
        if isinstance(value, float):
            if value < 0.01:
                print(f"    - {key}: {value:.2e}")
            else:
                print(f"    - {key}: {value:.4f}")
        else:
            print(f"    - {key}: {value}")
    
    print("\n" + "="*80)


def main():
    """Main visualization workflow."""
    print("\n" + "="*80)
    print("VISUALIZING OPTUNA TUNING RESULTS")
    print("="*80)
    
    # Load results
    results = load_optuna_results()
    
    if not results:
        print("No Optuna results found.")
        print("Run tuning first with: python tune.py --architecture <arch> --n-trials 100")
        return
    
    print(f"\nFound tuning results for {len(results)} architecture(s): {', '.join(sorted(results.keys()))}")
    
    # Print summary
    print_summary(results)
    
    # Create comprehensive plot
    create_comprehensive_plot(results, output_file="optuna_results_visualization.png")
    
    # Create detailed architecture plots
    print("\nGenerating detailed architecture plots...")
    create_detailed_architecture_plots(results)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - optuna_results_visualization.png (comprehensive overview)")
    print("  - optuna_results_<arch>_details.png (detailed per-architecture)")


if __name__ == "__main__":
    main()
