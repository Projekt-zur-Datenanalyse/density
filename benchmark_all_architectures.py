"""Benchmark all architectures with comprehensive analysis and plots."""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from analyze_data_ranges import analyze_data_ranges


class BenchmarkRunner:
    """Run benchmarks for all architectures."""
    
    def __init__(self, epochs: int = 100, learning_rate: float = 0.01):
        """Initialize benchmark runner.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.results = {}
        self.data_info = analyze_data_ranges()
        
    def run_architecture(self, architecture: str, name: str) -> dict:
        """Run training for a single architecture.
        
        Args:
            architecture: Architecture name (mlp, cnn, cnn_multiscale, gnn)
            name: Display name
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*80}")
        print(f"TRAINING: {name.upper()}")
        print(f"{'='*80}")
        
        # Run training using sys.executable to use current Python environment
        cmd = [
            sys.executable, "train.py",
            "--architecture", architecture,
            "--epochs", str(self.epochs),
            "--learning-rate", str(self.learning_rate),
            "--output-dir", f"./results_{architecture}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                print(f"ERROR running {name}:")
                print(result.stderr)
                return None
            
            # Load results
            results_dir = Path(f"./results_{architecture}")
            with open(results_dir / "test_results.json", 'r') as f:
                test_results = json.load(f)
            
            with open(results_dir / "training_history.json", 'r') as f:
                history = json.load(f)
            
            with open(results_dir / "model_config.json", 'r') as f:
                model_config = json.load(f)
            
            # Denormalize RMSE
            target_std = self.data_info['density']['std']
            test_rmse_denorm = test_results['test_rmse'] * target_std
            
            # Calculate metrics
            train_rmses = history['train_loss']
            val_rmses = history['val_loss']
            best_val_epoch = np.argmin(val_rmses) + 1
            best_val_rmse = min(val_rmses)
            
            best_val_rmse_denorm = best_val_rmse * target_std
            
            result_dict = {
                'name': name,
                'architecture': architecture,
                'test_rmse_norm': test_results['test_rmse'],
                'test_rmse_denorm': test_rmse_denorm,
                'best_val_rmse_norm': best_val_rmse,
                'best_val_rmse_denorm': best_val_rmse_denorm,
                'best_val_epoch': best_val_epoch,
                'train_history': train_rmses,
                'val_history': val_rmses,
                'num_parameters': model_config.get('num_parameters', 0),
                'model_config': model_config,
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
            ('gnn', 'Graph Neural Network'),
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
        
        print(f"\n{'Rank':<6} {'Architecture':<25} {'Test RMSE':<20} {'Best Val':<20} {'Parameters':<15} {'Best @ Epoch':<15}")
        print("-" * 100)
        
        for rank, (arch, result) in enumerate(sorted_archs, 1):
            print(f"{rank:<6} {result['name']:<25} "
                  f"{result['test_rmse_denorm']:<20.2f} kg/m³ "
                  f"{result['best_val_rmse_denorm']:<20.2f} kg/m³ "
                  f"{result['model_config'].get('total_parameters', 'N/A'):<15} "
                  f"{result['best_val_epoch']:<15}")
        
        # Performance analysis
        print("\n" + "=" * 100)
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
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Green, Blue, Orange, Red
        
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
        # Calculate parameters from model config or use placeholder
        params = []
        for _, result in sorted_archs:
            # Estimate based on architecture
            config = result['model_config']
            arch = config.get('architecture', 'mlp')
            
            if arch == 'mlp':
                # Simple MLP: input -> hidden -> output
                expansion = config.get('expansion_factor', 8)
                est_params = 4 * (4 * expansion) + (4 * expansion) * 1 + expansion + 1
            elif arch == 'cnn':
                # CNN: approximation based on channels
                channels = config.get('cnn_expansion_size', 8)
                kernel = config.get('cnn_kernel_size', 3)
                layers = config.get('cnn_num_layers', 4)
                est_params = layers * (channels * kernel + channels + 1)
            elif arch == 'cnn_multiscale':
                # Multi-scale CNN: multiple branch approximation
                channels = config.get('cnn_multiscale_base_channels', 16)
                scales = config.get('cnn_multiscale_num_scales', 3)
                layers = config.get('num_layers', 4)
                est_params = scales * layers * (channels * 9 + channels)
            elif arch == 'gnn':
                # GNN: node features * hidden * layers
                hidden = config.get('gnn_hidden_dim', 64)
                layers = config.get('gnn_num_layers', 3)
                est_params = 4 * hidden + layers * (hidden * hidden + hidden)
            else:
                est_params = 1000
            
            params.append(est_params)
        
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
        epochs_range = range(1, self.epochs + 1)
        
        for (arch, result), color in zip(sorted_archs, colors):
            ax4.plot(epochs_range, result['train_history'], 
                    label=f"{result['name']} (train)", 
                    color=color, linestyle='-', alpha=0.7, linewidth=2)
        
        ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax4.set_ylabel('RMSE (normalized)', fontsize=11, fontweight='bold')
        ax4.set_title('Training RMSE Trajectory', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(alpha=0.3)
        
        # 5. Validation History - RMSE Curves (Bottom Center)
        ax5 = plt.subplot(2, 3, 5)
        
        for (arch, result), color in zip(sorted_archs, colors):
            ax5.plot(epochs_range, result['val_history'], 
                    label=f"{result['name']} (val)", 
                    color=color, linestyle='--', alpha=0.7, linewidth=2, marker='o', markersize=3)
        
        # Add horizontal line for baseline
        baseline_rmse = self.data_info['density']['std']
        ax5.axhline(y=baseline_rmse, color='red', linestyle=':', linewidth=2, 
                   label=f'Naive Baseline ({baseline_rmse:.2f})')
        
        ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax5.set_ylabel('RMSE (normalized)', fontsize=11, fontweight='bold')
        ax5.set_title('Validation RMSE Trajectory', fontsize=12, fontweight='bold')
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
            summary_text += f"   Best @ Epoch: {result['best_val_epoch']}\n\n"
        
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
            json_results[arch] = {
                'name': result['name'],
                'architecture': result['architecture'],
                'test_rmse_norm': float(result['test_rmse_norm']),
                'test_rmse_denorm': float(result['test_rmse_denorm']),
                'best_val_rmse_norm': float(result['best_val_rmse_norm']),
                'best_val_rmse_denorm': float(result['best_val_rmse_denorm']),
                'best_val_epoch': int(result['best_val_epoch']),
                'num_parameters': int(result['model_config'].get('total_parameters', 0)),
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")


def main():
    """Main benchmark workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all architectures")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Output plot filename")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CHEMICAL DENSITY SURROGATE MODEL - ARCHITECTURE BENCHMARK")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    
    # Run benchmarks
    runner = BenchmarkRunner(epochs=args.epochs, learning_rate=args.learning_rate)
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
