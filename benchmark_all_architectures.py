"""Benchmark all architectures with comprehensive analysis and plots."""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from analyze_data_ranges import analyze_data_ranges


def find_latest_optuna_result(architecture: str) -> dict:
    """Find the latest optuna result for a given architecture.
    
    Args:
        architecture: Architecture name (mlp, cnn, cnn_multiscale, lightgbm)
    
    Returns:
        Dictionary with tuning metadata if found, empty dict if not found
    """
    # Create pattern that matches exact architecture name
    # e.g., "optuna_results_mlp_*" but not "optuna_results_cnn_multiscale_*" when looking for "cnn"
    if architecture == "cnn":
        # Special case: search for cnn but not cnn_multiscale
        optuna_dirs = []
        for p in sorted(Path(".").glob("optuna_results_cnn_*"), reverse=True):
            if "_cnn_" in p.name and "_multiscale_" not in p.name:
                optuna_dirs.append(p)
    elif architecture == "cnn_multiscale":
        optuna_dirs = sorted(Path(".").glob(f"optuna_results_cnn_multiscale_*"), reverse=True)
    else:
        pattern = f"optuna_results_{architecture}_*"
        optuna_dirs = sorted(Path(".").glob(pattern), reverse=True)
    
    if not optuna_dirs:
        return {}
    
    latest_dir = optuna_dirs[0]
    best_config_file = latest_dir / "best_config.json"
    optuna_summary_file = latest_dir / "optuna_summary.json"
    
    if not best_config_file.exists():
        return {}
    
    try:
        with open(best_config_file, 'r') as f:
            best_config = json.load(f)
        
        metadata = {
            'found': True,
            'directory': str(latest_dir),
            'best_config': best_config,
            'hyperparameters': best_config.get('hyperparameters', {}),
        }
        
        # Try to load optuna summary for additional metadata
        if optuna_summary_file.exists():
            with open(optuna_summary_file, 'r') as f:
                summary = json.load(f)
            metadata['n_trials'] = summary.get('n_trials', 0)
            metadata['best_trial_number'] = summary.get('best_trial', {}).get('number', 0)
        else:
            # Fall back to info from best_config
            trial_info = best_config.get('trial_info', {})
            metadata['n_trials'] = trial_info.get('n_trials', 0)
            metadata['best_trial_number'] = trial_info.get('best_trial_number', 0)
        
        return metadata
    except Exception as e:
        print(f"Warning: Could not load optuna results for {architecture}: {e}")
        return {}


class BenchmarkRunner:
    """Run benchmarks for all architectures."""
    
    def __init__(self, epochs: int = 100, learning_rate: float = 0.01, use_tuned_config: bool = False, seed: int = 46):
        """Initialize benchmark runner.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            use_tuned_config: Whether to use tuned hyperparameters from Optuna if available
            seed: Random seed for reproducibility (default: 46)
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_tuned_config = use_tuned_config
        self.seed = seed
        self.results = {}
        self.data_info = analyze_data_ranges()
        self.tuning_metadata = {}  # Store tuning info for each architecture
        
    def run_architecture(self, architecture: str, name: str) -> dict:
        """Run training for a single architecture.
        
        Args:
            architecture: Architecture name (mlp, cnn, cnn_multiscale)
            name: Display name
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*80}")
        print(f"TRAINING: {name.upper()}")
        print(f"{'='*80}")
        
        # Check for tuned config if requested
        tuned_config = {}
        if self.use_tuned_config:
            tuned_config = find_latest_optuna_result(architecture)
            if tuned_config.get('found'):
                print(f"[TUNED] Using configuration from {Path(tuned_config['directory']).name}")
                print(f"[TUNED] Best trial: #{tuned_config.get('best_trial_number', '?')} / {tuned_config.get('n_trials', '?')} trials")
                self.tuning_metadata[architecture] = tuned_config
            else:
                print(f"[CONFIG] Using default configuration from config.py")
        
        # Run training using sys.executable to use current Python environment
        cmd = [
            sys.executable, "train.py",
            "--architecture", architecture,
            "--epochs", str(self.epochs),
            "--learning-rate", str(self.learning_rate),
            "--seed", str(self.seed),
            "--output-dir", f"./results_{architecture}"
        ]
        
        # Add tuned hyperparameters to command if available
        if tuned_config.get('found') and tuned_config.get('hyperparameters'):
            hyperparams = tuned_config['hyperparameters']
            
            # Map hyperparameters to command line arguments
            # Includes both architecture-specific params and training config params
            # Note: Some flags in train.py are "disable" flags (--cnn-no-*), so we invert boolean logic
            param_map = {
                # Training configuration parameters (common to all architectures)
                'learning_rate': '--learning-rate',
                'batch_size': '--batch-size',
                'dropout_rate': '--dropout-rate',
                'weight_decay': '--weight-decay',
                'optimizer': '--optimizer',
                'lr_scheduler': '--lr-scheduler',
                'loss_fn': '--loss',
                
                # MLP-specific parameters
                'hidden_layer_dims': '--hidden-layer-dims',
                'use_swiglu': '--use-swiglu',
                
                # CNN-specific parameters
                'cnn_expansion_size': '--cnn-expansion-size',
                'cnn_num_layers': '--cnn-num-layers',
                'cnn_kernel_size': '--cnn-kernel-size',
                'cnn_use_batch_norm': '--cnn-no-batch-norm',  # Disable flag - inverted logic
                'cnn_use_residual': '--cnn-no-residual',      # Disable flag - inverted logic
                
                # Multi-Scale CNN parameters
                'cnn_multiscale_expansion_size': '--cnn-ms-expansion-size',
                'cnn_multiscale_num_scales': '--cnn-ms-num-scales',
                'cnn_multiscale_base_channels': '--cnn-ms-base-channels',
                
                # LightGBM-specific parameters
                'lgb_num_leaves': '--lgb-num-leaves',
                'lgb_learning_rate': '--lgb-learning-rate',
                'lgb_num_boost_round': '--lgb-num-boost-round',
                'lgb_max_depth': '--lgb-max-depth',
                'lgb_min_child_samples': '--lgb-min-child-samples',
                'lgb_subsample': '--lgb-subsample',
                'lgb_colsample_bytree': '--lgb-colsample-bytree',
                'lgb_reg_alpha': '--lgb-reg-alpha',
                'lgb_reg_lambda': '--lgb-reg-lambda',
                'lgb_boosting_type': '--lgb-boosting-type',
            }
            
            # Parameters with inverted boolean logic (disable flags)
            inverted_bool_params = {'cnn_use_batch_norm', 'cnn_use_residual'}
            
            for param, arg_name in param_map.items():
                if param in hyperparams:
                    value = hyperparams[param]
                    # Handle boolean flags
                    if isinstance(value, bool):
                        if param in inverted_bool_params:
                            # For disable flags, append if value is False (meaning disable)
                            if not value:
                                cmd.append(arg_name)
                        else:
                            # For regular enable flags, append if value is True
                            if value:
                                cmd.append(arg_name)
                    elif isinstance(value, list):
                        # For hidden_layer_dims, pass as space-separated values
                        cmd.extend([arg_name] + [str(v) for v in value])
                    else:
                        cmd.extend([arg_name, str(value)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                print(f"ERROR running {name}:")
                print(result.stderr)
                return None
            
            # Load results
            results_dir = Path(f"./results_{architecture}")
            with open(results_dir / "test_results.json", 'r') as f:
                test_results = json.load(f)
            
            # Try to load training history (may not exist for non-NN models like LightGBM)
            history_file = results_dir / "training_history.json"
            has_training_history = False
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                train_rmses = history.get('train_loss', [])
                val_rmses = history.get('val_loss', [])
                
                # Check if this is an NN model with real epoch history
                # NN models should have multiple epochs, LightGBM will have only 1
                if val_rmses and len(val_rmses) > 1:
                    best_val_epoch = np.argmin(val_rmses) + 1
                    best_val_rmse = min(val_rmses)
                    has_training_history = True
                elif val_rmses and len(val_rmses) == 1 and architecture != "lightgbm":
                    # Single epoch for non-LightGBM is unusual but treat it as having history
                    best_val_epoch = 1
                    best_val_rmse = val_rmses[0]
                    has_training_history = True
                else:
                    best_val_epoch = 0
                    best_val_rmse = float('inf')
            else:
                # For models without training history file
                train_rmses = []
                val_rmses = []
                best_val_epoch = 0  # No epoch information for non-NN models
                best_val_rmse = float('inf')
            
            with open(results_dir / "model_config.json", 'r') as f:
                model_config = json.load(f)
            
            # Use denormalized values from test_results if available, otherwise compute
            if 'test_rmse_denormalized' in test_results:
                test_rmse_denorm = test_results['test_rmse_denormalized']
            else:
                target_std = self.data_info['density']['std']
                test_rmse_denorm = test_results['test_rmse'] * target_std
            
            # If we have validation history, denormalize it
            if has_training_history and best_val_rmse != float('inf'):
                target_std = self.data_info['density']['std']
                best_val_rmse_denorm = best_val_rmse * target_std
            else:
                # For non-NN models (like LightGBM with single epoch), still denormalize the validation RMSE if available
                if best_val_rmse != float('inf') and best_val_rmse > 0:
                    target_std = self.data_info['density']['std']
                    best_val_rmse_denorm = best_val_rmse * target_std
                else:
                    best_val_rmse_denorm = test_rmse_denorm  # Fallback to test RMSE
            
            result_dict = {
                'name': name,
                'architecture': architecture,
                'test_rmse_norm': test_results['test_rmse'],
                'test_rmse_denorm': test_rmse_denorm,
                'best_val_rmse_norm': best_val_rmse if has_training_history else None,
                'best_val_rmse_denorm': best_val_rmse_denorm,
                'best_val_epoch': best_val_epoch,
                'has_training_history': has_training_history,  # Track whether this model has epoch-by-epoch history
                'train_history': train_rmses,
                'val_history': val_rmses,
                'num_parameters': model_config.get('num_parameters', 0),
                'model_config': model_config,
                'tuning_metadata': self.tuning_metadata.get(architecture, {}),  # Add tuning info
            }
            
            print(f"\n✓ {name} - Test RMSE: {test_rmse_denorm:.2f} kg/m³")
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Training {name} timed out")
            return None
        except Exception as e:
            print(f"ERROR running {name}: {e}")
            return None
    
    def run_all(self):
        """Run training for all architectures."""
        architectures = [
            ('mlp', 'Multi-Layer Perceptron'),
            ('cnn', 'Convolutional Neural Network'),
            ('cnn_multiscale', 'Multi-Scale CNN'),
            ('lightgbm', 'LightGBM Gradient Boosting'),
        ]
        
        for arch, name in architectures:
            result = self.run_architecture(arch, name)
            if result:
                self.results[arch] = result
    
    def generate_report(self):
        """Generate text report of results."""
        if not self.results:
            print("No results to report")
            return
        
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)
        
        # Sort by test RMSE (best first)
        sorted_archs = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_rmse_denorm'])
        
        print(f"\n{'Rank':<6} {'Architecture':<25} {'Test RMSE':<18} {'Val RMSE':<18} {'Parameters':<12} {'Val @ Epoch':<12}")
        print("-" * 100)
        
        for rank, (arch, result) in enumerate(sorted_archs, 1):
            # Format the Val RMSE column - show N/A for models without val_history, show actual val RMSE otherwise
            if result['has_training_history']:
                val_rmse_str = f"{result['best_val_rmse_denorm']:.2f} kg/m³"
                epoch_str = str(result['best_val_epoch'])
            elif result['val_history'] and len(result['val_history']) > 0:
                # Single-epoch model (like LightGBM) - still has validation RMSE
                val_rmse_str = f"{result['best_val_rmse_denorm']:.2f} kg/m³ (1 epoch)"
                epoch_str = "1"
            else:
                val_rmse_str = "N/A (no history)"
                epoch_str = "N/A"
            
            print(f"{rank:<6} {result['name']:<25} "
                  f"{result['test_rmse_denorm']:<15.2f} kg/m³ "
                  f"{val_rmse_str:<18} "
                  f"{result['model_config'].get('total_parameters', 'N/A'):<12} "
                  f"{epoch_str:<12}")
        
        # Print tuning information if available
        print("\n" + "=" * 100)
        print("HYPERPARAMETER TUNING INFORMATION")
        print("=" * 100)
        
        has_any_tuning = any(result.get('tuning_metadata', {}).get('found', False) for _, result in sorted_archs)
        
        if has_any_tuning:
            for arch, result in sorted_archs:
                tuning = result.get('tuning_metadata', {})
                if tuning.get('found'):
                    n_trials = tuning.get('n_trials', '?')
                    best_trial = tuning.get('best_trial_number', '?')
                    print(f"\n{result['name']}:")
                    print(f"  • Tuned with Optuna: {n_trials} trials")
                    print(f"  • Best trial: #{best_trial}")
                    print(f"  • Directory: {Path(tuning['directory']).name}")
                else:
                    print(f"\n{result['name']}:")
                    print(f"  • Configuration: Default (from config.py)")
        else:
            print("\nNo tuned configurations found. All models use default configuration from config.py.")
            print("To use tuned configurations, run: python benchmark_all_architectures.py --use-tuned-config")
        
        # Performance analysis
        print("\n" + "=" * 100)
        print("PERFORMANCE ANALYSIS & METRICS EXPLANATION")
        print("=" * 100)
        
        print(f"\nMetrics Terminology:")
        print(f"  [TRAIN RMSE] Average RMSE during training (tracks fitting to training data)")
        print(f"  [VAL RMSE]   Average RMSE on validation set (used to detect overfitting & select best model)")
        print(f"  [TEST RMSE]  Average RMSE on test set (true generalization performance on unseen data)")
        
        print(f"\nKey Insight:")
        print(f"  • Validation RMSE is used to checkpoint the best model during training")
        print(f"  • Test RMSE measures how well the model generalizes to completely unseen data")
        print(f"  • The 'Val RMSE' shown is the BEST validation RMSE achieved during training")
        print(f"  • This helps you understand if overfitting occurred (Val RMSE << Test RMSE)")
        
        print(f"\n" + "=" * 100)
        print("PERFORMANCE ANALYSIS")
        print("=" * 100)
        
        best_result = sorted_archs[0][1]
        worst_result = sorted_archs[-1][1]
        baseline_rmse = self.data_info['density']['std']
        
        print(f"\nData Reference:")
        print(f"  • Density range: {self.data_info['density']['min']:.2f} - {self.data_info['density']['max']:.2f} kg/m³")
        print(f"  • Density mean: {self.data_info['density']['mean']:.2f} kg/m³")
        print(f"  • Density std: {baseline_rmse:.2f} kg/m³")
        print(f"  • Naive baseline (predict mean): {baseline_rmse:.2f} kg/m³")
        
        print(f"\nBest Model: {best_result['name']}")
        print(f"  • Test RMSE: {best_result['test_rmse_denorm']:.2f} kg/m³")
        improvement_pct = ((baseline_rmse - best_result['test_rmse_denorm']) / baseline_rmse) * 100
        print(f"  • Improvement over baseline: {improvement_pct:.1f}%")
        print(f"  • As % of range: {(best_result['test_rmse_denorm'] / self.data_info['density']['range']) * 100:.1f}%")
        print(f"  • As % of std: {(best_result['test_rmse_denorm'] / baseline_rmse) * 100:.1f}%")
        print(f"  • As % of mean: {(best_result['test_rmse_denorm'] / self.data_info['density']['mean']) * 100:.1f}%")
        
        # Practical interpretation
        print(f"\nPractical Accuracy at Mean Density ({self.data_info['density']['mean']:.0f} kg/m³):")
        print(f"  • Prediction range: ±{best_result['test_rmse_denorm']:.2f} kg/m³")
        print(f"  • Error: ±{(best_result['test_rmse_denorm'] / self.data_info['density']['mean']) * 100:.1f}%")
        
        print(f"\nComparison:")
        for arch, result in sorted_archs:
            rmse_diff = result['test_rmse_denorm'] - best_result['test_rmse_denorm']
            pct_diff = (rmse_diff / best_result['test_rmse_denorm']) * 100
            print(f"  • {result['name']:<25}: {result['test_rmse_denorm']:>8.2f} kg/m³ ({pct_diff:+.1f}%)")
    
    def plot_results(self, output_file: str = "benchmark_results.png"):
        """Generate comprehensive plot of results.
        
        Args:
            output_file: Output filename for plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Sort by test RMSE
        sorted_archs = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_rmse_denorm'])
        
        fig = plt.figure(figsize=(16, 12))
        
        # Color scheme
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']  # Green, Blue, Orange, Red, Purple, Teal
        
        # Ensure we have enough colors
        num_archs = len(sorted_archs)
        while len(colors) < num_archs:
            colors.extend(['#95a5a6', '#34495e', '#e67e22', '#c0392b'])  # Additional colors
        
        # 1. Test RMSE Comparison (Top Left)
        ax1 = plt.subplot(2, 3, 1)
        names = [result['name'] for _, result in sorted_archs]
        test_rmses = [result['test_rmse_denorm'] for _, result in sorted_archs]
        bars1 = ax1.bar(range(len(names)), test_rmses, color=colors)
        ax1.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
        ax1.set_title('Test RMSE by Architecture', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=15, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, rmse) in enumerate(zip(bars1, test_rmses)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{rmse:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Best Validation RMSE Comparison (Top Middle)
        ax2 = plt.subplot(2, 3, 2)
        best_val_rmses = [result['best_val_rmse_denorm'] for _, result in sorted_archs]
        bars2 = ax2.bar(range(len(names)), best_val_rmses, color=colors)
        ax2.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
        ax2.set_title('Best Validation RMSE', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (bar, rmse) in enumerate(zip(bars2, best_val_rmses)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{rmse:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Parameters Comparison (Top Right)
        ax3 = plt.subplot(2, 3, 3)
        # Get actual parameter counts from trained models
        params = []
        for arch, result in sorted_archs:
            config = result['model_config']
            actual_arch = config.get('architecture', 'mlp')
            
            # Try to load actual parameter count from checkpoint
            actual_params = None
            results_dir = Path(f"./results_{actual_arch}")
            checkpoint_path = results_dir / 'checkpoints' / 'best_model.pt'
            
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        actual_params = sum(p.numel() for p in state_dict.values())
                except Exception as e:
                    actual_params = None
            
            # Fallback: use config-based estimation if checkpoint not available
            if actual_params is None:
                if actual_arch == 'mlp':
                    expansion = config.get('expansion_factor', 4)
                    num_layers = config.get('num_layers', 2)
                    hidden_dims = config.get('hidden_layer_dims')
                    if hidden_dims:
                        # Calculate from explicit hidden dimensions
                        dims = [4] + hidden_dims + [1]
                        actual_params = sum(dims[i] * dims[i+1] + dims[i+1] for i in range(len(dims)-1))
                    else:
                        # Calculate from expansion factor
                        hidden_dim = 4 * expansion
                        dims = [4] + [hidden_dim] * num_layers + [1]
                        actual_params = sum(dims[i] * dims[i+1] + dims[i+1] for i in range(len(dims)-1))
                elif actual_arch == 'cnn':
                    expansion_size = config.get('cnn_expansion_size', 4)
                    num_layers = config.get('cnn_num_layers', 2)
                    kernel_size = config.get('cnn_kernel_size', 3)
                    # Rough estimate: expand layer + conv layers + output head
                    expanded_size = expansion_size * expansion_size
                    params_expand = 4 * expanded_size + expanded_size
                    # Conv layers with channel progression: 1 -> 16 -> 32 -> 64 -> 128
                    channels = [1, 16, 32, 64, 128][:num_layers+1]
                    params_conv = sum(
                        channels[i] * channels[i+1] * (kernel_size**2) + channels[i+1]
                        for i in range(len(channels)-1)
                    )
                    # Output head: channels[-1] -> 64 -> 1
                    params_head = channels[-1] * 64 + 64 + 64 * 1 + 1
                    actual_params = params_expand + params_conv + params_head
                elif actual_arch == 'cnn_multiscale':
                    expansion_size = config.get('cnn_multiscale_expansion_size', 20)
                    num_scales = config.get('cnn_multiscale_num_scales', 3)
                    base_channels = config.get('cnn_multiscale_base_channels', 8)
                    # Rough estimate: expand layer + 3 branches (each with 2 conv layers) + output head
                    expanded_size = expansion_size * expansion_size
                    params_expand = 4 * expanded_size + expanded_size
                    # Each branch: Conv(1->base) -> Conv(base->base) x2, then pool
                    params_per_branch = (1 * base_channels * 9 + base_channels) * 2  # Two conv layers per branch
                    params_branches = num_scales * params_per_branch
                    # Output head: base_channels * num_scales -> 64 -> 1
                    params_head = (base_channels * num_scales) * 64 + 64 + 64 * 1 + 1
                    actual_params = params_expand + params_branches + params_head
                elif actual_arch == 'lightgbm':
                    # LightGBM: tree complexity metric (not traditional parameters)
                    num_leaves = config.get('lgb_num_leaves', 31)
                    num_rounds = config.get('lgb_num_boost_round', 100)
                    actual_params = num_leaves * num_rounds
                else:
                    actual_params = 1000
            
            params.append(actual_params)
        
        bars3 = ax3.bar(range(len(names)), params, color=colors)
        ax3.set_ylabel('Estimated Parameters', fontsize=11, fontweight='bold')
        ax3.set_title('Model Complexity', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=15, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        for i, (bar, param) in enumerate(zip(bars3, params)):
            if param > 0:
                # Format with K for thousands
                if param >= 1000:
                    label = f'{param/1000:.1f}K'
                else:
                    label = f'{param:.0f}'
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Training History - RMSE Curves (Bottom Left & Center)
        ax4 = plt.subplot(2, 3, 4)
        
        has_train_history = False
        for (arch, result), color in zip(sorted_archs, colors):
            # Only plot if model has training history (exclude LightGBM)
            if result['has_training_history'] and result['train_history'] and len(result['train_history']) > 1:
                has_train_history = True
                epochs_range = range(1, len(result['train_history']) + 1)
                ax4.plot(epochs_range, result['train_history'], 
                        label=f"{result['name']} (train)", 
                        color=color, linestyle='-', alpha=0.7, linewidth=2)
        
        if not has_train_history:
            ax4.text(0.5, 0.5, 'No epoch-by-epoch training history\n(Tree-based models use different training)',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        
        ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax4.set_ylabel('RMSE (normalized)', fontsize=11, fontweight='bold')
        ax4.set_title('Training RMSE Trajectory', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(alpha=0.3)
        
        # 5. Validation History - RMSE Curves (Bottom Center)
        ax5 = plt.subplot(2, 3, 5)
        
        # Get target std for denormalization
        target_std = self.data_info['density']['std']
        
        has_val_history = False
        for (arch, result), color in zip(sorted_archs, colors):
            # Only plot if model has validation history (exclude LightGBM and non-NN models)
            if result['has_training_history'] and result['val_history'] and len(result['val_history']) > 1:
                has_val_history = True
                epochs_range = range(1, len(result['val_history']) + 1)
                # Denormalize validation RMSE for plotting
                val_history_denorm = [v * target_std for v in result['val_history']]
                ax5.plot(epochs_range, val_history_denorm, 
                        label=f"{result['name']} (val)", 
                        color=color, linestyle='--', alpha=0.7, linewidth=2, marker='o', markersize=3)
        
        if not has_val_history:
            ax5.text(0.5, 0.5, 'No epoch-by-epoch validation history\n(Tree-based models use different training)',
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
        
        # Add horizontal line for baseline
        baseline_rmse = self.data_info['density']['std']
        ax5.axhline(y=baseline_rmse, color='red', linestyle=':', linewidth=2, 
                   label=f'Naive Baseline ({baseline_rmse:.2f})')
        
        ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax5.set_ylabel('RMSE (kg/m³)', fontsize=11, fontweight='bold')
        ax5.set_title('Validation RMSE Trajectory (Denormalized)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9, loc='best')
        ax5.grid(alpha=0.3)
        
        # 6. Performance Metrics (Bottom Right)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create text summary
        summary_text = "PERFORMANCE SUMMARY\n" + "="*50 + "\n\n"
        
        for rank, (arch, result) in enumerate(sorted_archs, 1):
            improvement = ((baseline_rmse - result['test_rmse_denorm']) / baseline_rmse) * 100
            summary_text += f"{rank}. {result['name']}\n"
            summary_text += f"   Test RMSE: {result['test_rmse_denorm']:.2f} kg/m³\n"
            summary_text += f"   Improvement: {improvement:.1f}%\n"
            if result['has_training_history']:
                summary_text += f"   Best @ Epoch: {result['best_val_epoch']}\n"
            else:
                summary_text += f"   Best @ Epoch: N/A (non-NN)\n"
            summary_text += "\n"
        
        summary_text += "\nDATA REFERENCE:\n" + "-"*50 + "\n"
        summary_text += f"Density Range: {self.data_info['density']['min']:.0f} - {self.data_info['density']['max']:.0f} kg/m³\n"
        summary_text += f"Mean: {self.data_info['density']['mean']:.0f} kg/m³\n"
        summary_text += f"Std: {baseline_rmse:.2f} kg/m³\n"
        summary_text += f"Baseline RMSE: {baseline_rmse:.2f} kg/m³\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_file}")
        plt.close()
    
    def save_results_json(self, output_file: str = "benchmark_results.json"):
        """Save results to JSON file.
        
        Args:
            output_file: Output filename
        """
        # Convert results to JSON-serializable format
        json_results = {}
        for arch, result in self.results.items():
            # Handle None values for non-NN models (like LightGBM)
            best_val_rmse_norm = result['best_val_rmse_norm']
            best_val_rmse_denorm = result['best_val_rmse_denorm']
            
            json_results[arch] = {
                'name': result['name'],
                'architecture': result['architecture'],
                'test_rmse_norm': float(result['test_rmse_norm']),
                'test_rmse_denorm': float(result['test_rmse_denorm']),
                'best_val_rmse_norm': float(best_val_rmse_norm) if best_val_rmse_norm is not None else None,
                'best_val_rmse_denorm': float(best_val_rmse_denorm) if best_val_rmse_denorm is not None else None,
                'best_val_epoch': int(result['best_val_epoch']) if result['best_val_epoch'] else None,
                'num_parameters': int(result['model_config'].get('total_parameters', 0)),
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"[OK] Results saved to {output_file}")


def main():
    """Main benchmark workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all architectures")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=46, help="Random seed for reproducibility (default: 46)")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Output plot filename")
    parser.add_argument("--use-tuned-config", action="store_true", help="Use tuned hyperparameters from Optuna if available (falls back to config.py defaults)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CHEMICAL DENSITY SURROGATE MODEL - ARCHITECTURE BENCHMARK")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Random Seed: {args.seed}")
    if args.use_tuned_config:
        print(f"Hyperparameters: Using Optuna-tuned configs if available")
    else:
        print(f"Hyperparameters: Using default config.py")
    
    # Run benchmarks
    runner = BenchmarkRunner(epochs=args.epochs, learning_rate=args.learning_rate, use_tuned_config=args.use_tuned_config, seed=args.seed)
    runner.run_all()
    
    # Generate outputs
    runner.generate_report()
    runner.plot_results(output_file=args.output)
    runner.save_results_json(output_file="benchmark_results.json")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
