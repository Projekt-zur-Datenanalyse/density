#!/usr/bin/env python
"""
Analyze and visualize Deep Ensemble results.

This script loads a trained deep ensemble, generates predictions with uncertainty,
and creates comprehensive visualizations showing:
- Individual model performance
- Ensemble vs individual model accuracy
- Prediction vs true values with uncertainty bands
- Error distribution
- Uncertainty vs error relationship
- Uncertainty vs true value relationship
- Per-model accuracy and uncertainty metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_models_and_predict(
    results_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[np.ndarray]]:
    """Load ensemble results and generate predictions from saved checkpoints.
    
    Args:
        results_dir: Path to ensemble results directory
        device: Device for inference
        
    Returns:
        metadata, test_targets, ensemble_predictions, individual_predictions_list
    """
    # Load metadata
    results_file = results_dir / "tuned_ensemble_results.json"
    configs_file = results_dir / "ensemble_configs_used.json"
    
    with open(results_file) as f:
        results = json.load(f)
    
    with open(configs_file) as f:
        configs_used = json.load(f)
    
    # Determine number of models
    n_models = results['n_models']
    
    print(f"Loading {n_models} models from checkpoints...")
    
    # Load test data to get targets
    from deep_ensemble import load_data_with_fixed_test_split, set_all_seeds
    
    _, _, test_loader, _, _, norm_stats = load_data_with_fixed_test_split(
        data_dir=".",
        master_seed=46,
        train_val_seed=46,
        batch_size=32,
    )
    
    # Collect all test targets
    all_targets = []
    for X_batch, y_batch in test_loader:
        all_targets.append(y_batch.cpu().numpy())
    all_targets = np.concatenate(all_targets).squeeze()
    
    # Denormalize targets
    target_mean = norm_stats['target_mean']
    target_std = norm_stats['target_std']
    test_targets_denorm = all_targets * target_std + target_mean
    
    # Load models and generate predictions
    individual_predictions = []
    
    for model_idx in range(1, n_models + 1):
        checkpoint_dir = results_dir / f"checkpoints_model_{model_idx}"
        checkpoint_file = checkpoint_dir / "best_model.pt"
        
        if not checkpoint_file.exists():
            print(f"Warning: Checkpoint for model {model_idx} not found")
            continue
        
        # Get config for this model
        config = configs_used[model_idx - 1]
        architecture = config.get('architecture', 'mlp')
        model_seed = config['seed']
        
        # Set seed for reproducibility
        set_all_seeds(model_seed)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location=device)
            
            # Check if it's a wrapper dict with 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Get hyperparameters
            hyperparams = config.get('params', {})
            
            # If params are missing, try to load from Optuna results
            if not hyperparams:
                hyperparams = _load_params_from_optuna(
                    config.get('trial_number'),
                    architecture
                )
            
            # Create model from hyperparameters
            from optuna_trainable import create_model_from_hyperparams
            model_obj = create_model_from_hyperparams(
                hyperparams,
                architecture,
                device=device
            )
            
            # Load state dict
            if isinstance(model_obj, torch.nn.Module):
                model_obj.load_state_dict(state_dict)
                model = model_obj
            else:
                # Non-torch model (e.g., LightGBM)
                model = model_obj
                
        except Exception as e:
            print(f"Error loading model {model_idx}: {e}")
            continue
        
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.eval()
        
        # Generate predictions
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                if isinstance(model, torch.nn.Module):
                    preds = model(X_batch).cpu().numpy()
                else:
                    # LightGBM or similar
                    preds = model.predict(X_batch.cpu().numpy()).reshape(-1, 1)
                all_preds.append(preds)
        
        preds = np.concatenate(all_preds).squeeze()
        # Denormalize
        preds_denorm = preds * target_std + target_mean
        individual_predictions.append(preds_denorm)
        
        print(f"  Model {model_idx}: loaded and inferred")
    
    # Stack predictions
    if not individual_predictions:
        raise RuntimeError("No models were successfully loaded!")
    
    individual_predictions = np.array(individual_predictions)  # Shape: (n_models, n_samples)
    
    # Compute ensemble predictions
    ensemble_predictions = np.mean(individual_predictions, axis=0)
    
    return results, test_targets_denorm, ensemble_predictions, individual_predictions


def _load_params_from_optuna(trial_number: int, architecture: str) -> Dict[str, Any]:
    """Load hyperparameters from Optuna results directory.
    
    Args:
        trial_number: Trial number to load
        architecture: Architecture name (mlp, cnn, cnn_multiscale, lightgbm)
        
    Returns:
        Dictionary of hyperparameters
    """
    from pathlib import Path
    import json
    
    # Find the most recent Optuna results directory for this architecture
    workspace_dir = Path(".")
    optuna_dirs = sorted(workspace_dir.glob(f"optuna_results_{architecture}_*"), reverse=True)
    
    if not optuna_dirs:
        print(f"Warning: No Optuna results found for architecture {architecture}")
        return {}
    
    # Try to load the config
    optuna_dir = optuna_dirs[0]
    configs_dir = optuna_dir / "configs"
    
    # Find the config file with matching rank
    for config_file in sorted(configs_dir.glob("config_rank_*.json")):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
                if cfg.get("trial_number") == trial_number:
                    return cfg.get("params", {})
        except:
            pass
    
    print(f"Warning: Could not find trial {trial_number} in Optuna results")
    return {}
    
    # Compute ensemble predictions
    ensemble_predictions = np.mean(individual_predictions, axis=0)
    
    return results, test_targets_denorm, ensemble_predictions, individual_predictions


def create_accuracy_comparison_plot(
    results: Dict,
    ensemble_predictions: np.ndarray,
    individual_predictions: np.ndarray,
    test_targets: np.ndarray,
    output_dir: Path
) -> None:
    """Create accuracy comparison bar plot.
    
    Args:
        results: Results dictionary
        ensemble_predictions: Ensemble predictions
        individual_predictions: Individual model predictions
        test_targets: True values
        output_dir: Output directory
    """
    # Compute RMSEs
    individual_rmses = results['individual_rmses']
    ensemble_rmse = results['ensemble_test_rmse_denorm']
    mean_individual_rmse = np.mean(individual_rmses)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Individual vs Ensemble
    model_labels = [f"Model {i+1}" for i in range(len(individual_rmses))]
    model_labels.append("Ensemble")
    rmses = individual_rmses + [ensemble_rmse]
    colors = ['lightblue'] * len(individual_rmses) + ['darkgreen']
    
    bars = ax1.bar(model_labels, rmses, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=mean_individual_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_individual_rmse:.2f}')
    ax1.set_ylabel('RMSE (kg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title('Test RMSE: Individual Models vs Ensemble', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Improvement metrics
    improvement = mean_individual_rmse - ensemble_rmse
    improvement_pct = (improvement / mean_individual_rmse) * 100
    
    metrics = ['Mean RMSE', 'Ensemble RMSE', 'Improvement']
    values = [mean_individual_rmse, ensemble_rmse, improvement]
    colors2 = ['lightblue', 'darkgreen', 'gold']
    
    bars2 = ax2.bar(metrics, values, color=colors2, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('RMSE (kg/m³)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Ensemble Improvement: {improvement_pct:.1f}%', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved: 01_accuracy_comparison.png")
    plt.close()


def create_prediction_vs_true_plot(
    ensemble_predictions: np.ndarray,
    individual_predictions: np.ndarray,
    test_targets: np.ndarray,
    output_dir: Path
) -> None:
    """Create prediction vs true values plot with uncertainty bands.
    
    Args:
        ensemble_predictions: Ensemble predictions
        individual_predictions: Individual model predictions
        test_targets: True values
        output_dir: Output directory
    """
    # Sort by true values for better visualization
    sorted_indices = np.argsort(test_targets)
    true_sorted = test_targets[sorted_indices]
    ensemble_sorted = ensemble_predictions[sorted_indices]
    individual_sorted = individual_predictions[:, sorted_indices]
    
    # Compute uncertainty
    uncertainty = np.std(individual_sorted, axis=0)
    
    x = np.arange(len(true_sorted))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot uncertainty band (min-max range)
    min_preds = np.min(individual_sorted, axis=0)
    max_preds = np.max(individual_sorted, axis=0)
    ax.fill_between(x, min_preds, max_preds, alpha=0.2, color='blue', label='Min-Max range')
    
    # Plot std bands around ensemble
    ax.fill_between(x, ensemble_sorted - uncertainty, ensemble_sorted + uncertainty,
                    alpha=0.3, color='orange', label='±1 Std (Uncertainty)')
    
    # Plot true values
    ax.plot(x, true_sorted, 'g-', linewidth=2.5, label='True Values', marker='o', markersize=3, markevery=max(1, len(x)//50))
    
    # Plot ensemble predictions
    ax.plot(x, ensemble_sorted, 'r-', linewidth=2.5, label='Ensemble Prediction', marker='s', markersize=3, markevery=max(1, len(x)//50))
    
    # Plot individual predictions (lighter)
    for i, preds in enumerate(individual_sorted):
        ax.plot(x, preds, '-', alpha=0.2, linewidth=1, color='blue')
    
    ax.set_xlabel('Test Sample (sorted by true value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density (kg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Predictions vs True Values with Uncertainty Bands', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_predictions_vs_true.png", dpi=300, bbox_inches='tight')
    print("Saved: 02_predictions_vs_true.png")
    plt.close()


def create_error_distribution_plot(
    ensemble_predictions: np.ndarray,
    individual_predictions: np.ndarray,
    test_targets: np.ndarray,
    output_dir: Path
) -> None:
    """Create error distribution plots.
    
    Args:
        ensemble_predictions: Ensemble predictions
        individual_predictions: Individual model predictions
        test_targets: True values
        output_dir: Output directory
    """
    # Compute errors
    ensemble_errors = ensemble_predictions - test_targets
    individual_errors = individual_predictions - test_targets.reshape(1, -1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Ensemble error distribution
    ax = axes[0, 0]
    ax.hist(ensemble_errors, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(np.mean(ensemble_errors), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ensemble_errors):.2f}')
    ax.set_xlabel('Error (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Ensemble Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Individual models error distributions
    ax = axes[0, 1]
    for i in range(individual_errors.shape[0]):
        ax.hist(individual_errors[i], bins=25, alpha=0.5, label=f'Model {i+1}')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Error (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Individual Models Error Distributions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Error magnitude comparison (box plot)
    ax = axes[1, 0]
    error_magnitudes = [np.abs(individual_errors[i]) for i in range(individual_errors.shape[0])]
    error_magnitudes.append(np.abs(ensemble_errors))
    labels = [f'M{i+1}' for i in range(individual_errors.shape[0])] + ['Ensemble']
    bp = ax.boxplot(error_magnitudes, labels=labels, patch_artist=True)
    
    # Color boxes
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor('lightblue' if 'M' in label else 'darkgreen')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Absolute Error (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_title('Error Magnitude Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative error distribution
    ax = axes[1, 1]
    ensemble_abs_errors = np.abs(ensemble_errors)
    sorted_errors = np.sort(ensemble_abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    ax.plot(sorted_errors, cumulative, 'darkgreen', linewidth=2.5, label='Ensemble')
    
    for i in range(min(3, individual_errors.shape[0])):  # Plot first 3 for clarity
        ind_abs_errors = np.abs(individual_errors[i])
        sorted_ind_errors = np.sort(ind_abs_errors)
        cumulative_ind = np.arange(1, len(sorted_ind_errors) + 1) / len(sorted_ind_errors) * 100
        ax.plot(sorted_ind_errors, cumulative_ind, '--', alpha=0.6, linewidth=1.5, label=f'Model {i+1}')
    
    ax.set_xlabel('Absolute Error (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_error_distributions.png", dpi=300, bbox_inches='tight')
    print("Saved: 03_error_distributions.png")
    plt.close()


def create_uncertainty_analysis_plots(
    ensemble_predictions: np.ndarray,
    individual_predictions: np.ndarray,
    test_targets: np.ndarray,
    output_dir: Path
) -> None:
    """Create uncertainty vs error and uncertainty vs true value plots.
    
    Args:
        ensemble_predictions: Ensemble predictions
        individual_predictions: Individual model predictions
        test_targets: True values
        output_dir: Output directory
    """
    # Compute metrics
    ensemble_errors = np.abs(ensemble_predictions - test_targets)
    uncertainty = np.std(individual_predictions, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot 1: Uncertainty vs Absolute Error (scatter)
    ax = axes[0, 0]
    scatter = ax.scatter(uncertainty, ensemble_errors, c=test_targets, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add correlation
    corr = np.corrcoef(uncertainty, ensemble_errors)[0, 1]
    ax.set_xlabel('Prediction Uncertainty (Std)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Error (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_title(f'Uncertainty vs Error (corr: {corr:.3f})', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('True Value (kg/m³)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(uncertainty, ensemble_errors, 1)
    p = np.poly1d(z)
    ax.plot(uncertainty, p(uncertainty), "r--", linewidth=2, alpha=0.8, label='Trend')
    ax.legend(fontsize=10)
    
    # Plot 2: Uncertainty vs True Value (scatter)
    ax = axes[0, 1]
    scatter = ax.scatter(test_targets, uncertainty, c=ensemble_errors, cmap='Reds',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('True Value (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Uncertainty (Std)', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty vs True Value', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error (kg/m³)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty histogram
    ax = axes[1, 0]
    ax.hist(uncertainty, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(uncertainty), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(uncertainty):.4f}')
    ax.axvline(np.median(uncertainty), color='orange', linestyle='--', linewidth=2,
              label=f'Median: {np.median(uncertainty):.4f}')
    ax.set_xlabel('Uncertainty (Std)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Residuals vs Uncertainty colored by true value
    ax = axes[1, 1]
    ensemble_signed_errors = ensemble_predictions - test_targets
    scatter = ax.scatter(uncertainty, ensemble_signed_errors, c=test_targets, cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Prediction Uncertainty (Std)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Signed Error (Predicted - True)', fontsize=11, fontweight='bold')
    ax.set_title('Residuals vs Uncertainty', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('True Value (kg/m³)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_uncertainty_analysis.png", dpi=300, bbox_inches='tight')
    print("Saved: 04_uncertainty_analysis.png")
    plt.close()


def create_model_comparison_plot(
    individual_predictions: np.ndarray,
    test_targets: np.ndarray,
    results: Dict,
    output_dir: Path
) -> None:
    """Create detailed model comparison plot.
    
    Args:
        individual_predictions: Individual model predictions
        test_targets: True values
        results: Results dictionary
        output_dir: Output directory
    """
    n_models = individual_predictions.shape[0]
    
    # Compute per-model metrics
    rmses = results['individual_rmses']
    maes = results['individual_maes']
    
    # Compute additional metrics
    r_squared_scores = []
    biases = []
    for preds in individual_predictions:
        errors = preds - test_targets
        bias = np.mean(errors)
        biases.append(bias)
        
        # R-squared
        ss_res = np.sum((errors) ** 2)
        ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r_squared_scores.append(r2)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RMSE per model
    ax = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_models))
    bars = ax.bar(range(1, n_models + 1), rmses, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(np.mean(rmses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmses):.2f}')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_title('RMSE per Model', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rmse:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: MAE per model
    ax = axes[0, 1]
    bars = ax.bar(range(1, n_models + 1), maes, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(np.mean(maes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(maes):.2f}')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_title('MAE per Model', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mae:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: R-squared per model
    ax = axes[1, 0]
    bars = ax.bar(range(1, n_models + 1), r_squared_scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(np.mean(r_squared_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r_squared_scores):.3f}')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax.set_title('R² Score per Model', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, r2 in zip(bars, r_squared_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Bias per model
    ax = axes[1, 1]
    colors_bias = ['red' if b < 0 else 'green' for b in biases]
    bars = ax.bar(range(1, n_models + 1), biases, color=colors_bias, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax.axhline(np.mean(biases), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(biases):.2f}')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Bias (kg/m³)', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Bias per Model', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, bias in zip(bars, biases):
        height = bar.get_height()
        y_pos = height if height > 0 else height
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{bias:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_model_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved: 05_model_comparison.png")
    plt.close()


def create_summary_report(
    results: Dict,
    ensemble_predictions: np.ndarray,
    individual_predictions: np.ndarray,
    test_targets: np.ndarray,
    output_dir: Path
) -> None:
    """Create a summary report.
    
    Args:
        results: Results dictionary
        ensemble_predictions: Ensemble predictions
        individual_predictions: Individual model predictions
        test_targets: True values
        output_dir: Output directory
    """
    ensemble_errors = np.abs(ensemble_predictions - test_targets)
    uncertainty = np.std(individual_predictions, axis=0)
    
    report = f"""
================================================================================
DEEP ENSEMBLE ANALYSIS REPORT
================================================================================

ENSEMBLE PERFORMANCE
--------------------
Number of Models: {results['n_models']}
Ensemble Test RMSE: {results['ensemble_test_rmse_denorm']:.4f} kg/m³
Ensemble Test MAE: {results['ensemble_test_mae_denorm']:.4f} kg/m³
Mean Model RMSE: {results['mean_test_rmse_denorm']:.4f} kg/m³
Std Model RMSE: {results['std_test_rmse_denorm']:.4f} kg/m³
Mean Model MAE: {results['mean_test_mae_denorm']:.4f} kg/m³

Improvement (Ensemble vs Mean): {results['mean_test_rmse_denorm'] - results['ensemble_test_rmse_denorm']:.4f} kg/m³
Improvement (%): {((results['mean_test_rmse_denorm'] - results['ensemble_test_rmse_denorm']) / results['mean_test_rmse_denorm'] * 100):.2f}%

UNCERTAINTY METRICS
-------------------
Mean Uncertainty: {results['mean_uncertainty']:.6f}
Std Uncertainty: {results['std_uncertainty']:.6f}

INDIVIDUAL MODEL PERFORMANCE
-----------------------------
"""
    
    for i, rmse in enumerate(results['individual_rmses'], 1):
        mae = results['individual_maes'][i-1]
        report += f"Model {i}: RMSE={rmse:.4f} kg/m³, MAE={mae:.4f} kg/m³\n"
    
    report += f"""
ERROR STATISTICS
----------------
Mean Error: {np.mean(ensemble_errors):.4f} kg/m³
Median Error: {np.median(ensemble_errors):.4f} kg/m³
Std Error: {np.std(ensemble_errors):.4f} kg/m³
Min Error: {np.min(ensemble_errors):.4f} kg/m³
Max Error: {np.max(ensemble_errors):.4f} kg/m³
95th Percentile Error: {np.percentile(ensemble_errors, 95):.4f} kg/m³

UNCERTAINTY STATISTICS
----------------------
Mean Uncertainty: {np.mean(uncertainty):.6f}
Median Uncertainty: {np.median(uncertainty):.6f}
Min Uncertainty: {np.min(uncertainty):.6f}
Max Uncertainty: {np.max(uncertainty):.6f}
95th Percentile Uncertainty: {np.percentile(uncertainty, 95):.6f}

CORRELATION ANALYSIS
--------------------
Uncertainty vs Error Correlation: {np.corrcoef(uncertainty, ensemble_errors)[0, 1]:.4f}
(Higher value indicates uncertainty is a better predictor of error)

SUMMARY
-------
The ensemble successfully combines {results['n_models']} models with an improvement of 
{((results['mean_test_rmse_denorm'] - results['ensemble_test_rmse_denorm']) / results['mean_test_rmse_denorm'] * 100):.2f}% over the mean individual model performance.

The uncertainty estimates have a correlation of {np.corrcoef(uncertainty, ensemble_errors)[0, 1]:.4f} with prediction
errors, indicating {"good" if abs(np.corrcoef(uncertainty, ensemble_errors)[0, 1]) > 0.5 else "moderate"} reliability of uncertainty quantification.

================================================================================
"""
    
    # Save report
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("Saved: analysis_report.txt")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Deep Ensemble results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze tuned ensemble results
  python analyze_ensemble.py --results-dir tuned_ensemble_results
  
  # Analyze with custom output
  python analyze_ensemble.py --results-dir tuned_ensemble_results --output-dir ensemble_analysis
  
  # Use CPU for inference
  python analyze_ensemble.py --results-dir tuned_ensemble_results --device cpu
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to ensemble results directory (containing tuned_ensemble_results.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ensemble_analysis",
        help="Output directory for visualizations (default: ensemble_analysis)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )
    
    args = parser.parse_args()
    
    # Setup
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DEEP ENSEMBLE ANALYSIS")
    print(f"{'='*80}")
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load models and generate predictions
    print("Loading ensemble and generating predictions...\n")
    results, test_targets, ensemble_predictions, individual_predictions = load_models_and_predict(
        results_dir,
        device=args.device
    )
    
    # Create visualizations
    print("\nGenerating visualizations...\n")
    
    create_accuracy_comparison_plot(
        results, ensemble_predictions, individual_predictions, test_targets, output_dir
    )
    
    create_prediction_vs_true_plot(
        ensemble_predictions, individual_predictions, test_targets, output_dir
    )
    
    create_error_distribution_plot(
        ensemble_predictions, individual_predictions, test_targets, output_dir
    )
    
    create_uncertainty_analysis_plots(
        ensemble_predictions, individual_predictions, test_targets, output_dir
    )
    
    create_model_comparison_plot(
        individual_predictions, test_targets, results, output_dir
    )
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(
        results, ensemble_predictions, individual_predictions, test_targets, output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Outputs saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
