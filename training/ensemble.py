"""
Deep Ensemble training for uncertainty-aware predictions.

This module provides ensemble training with:
- Multiple models with different seeds (same architecture)
- Multi-architecture ensembles
- Fixed test split across all models (for fair comparison)
- Variable train/val splits (for diversity)
- Uncertainty quantification via prediction variance
- Comprehensive evaluation and visualization
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from core import TrainingConfig, DataLoader, Trainer, set_seed, get_device
from models import create_model, AVAILABLE_ARCHITECTURES


class EnsembleTrainer:
    """Deep Ensemble trainer for uncertainty-aware predictions.
    
    Trains multiple models and combines their predictions for:
    - Improved accuracy through averaging
    - Uncertainty quantification via prediction variance
    
    Supports:
    - Same-architecture ensembles (multiple seeds)
    - Multi-architecture ensembles
    - Tuned configurations from Optuna
    
    Example:
        # Same-architecture ensemble
        trainer = EnsembleTrainer(
            architecture="mlp",
            n_models=5,
        )
        results = trainer.run()
        
        # Multi-architecture ensemble  
        trainer = EnsembleTrainer(
            architectures=["mlp", "cnn", "cnn_multiscale"],
        )
        results = trainer.run()
    """
    
    def __init__(
        self,
        architecture: Optional[str] = None,
        architectures: Optional[List[str]] = None,
        n_models: int = 5,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[TrainingConfig] = None,
        tuned_config_dir: Optional[str] = None,
        output_dir: str = "results_ensemble",
        master_seed: int = 46,
    ):
        """Initialize the ensemble trainer.
        
        Args:
            architecture: Single architecture for same-architecture ensemble
            architectures: List of architectures for multi-architecture ensemble
            n_models: Number of models per architecture
            model_config: Model-specific configuration
            training_config: Training configuration
            tuned_config_dir: Path to Optuna results for tuned configs
            output_dir: Directory to save results
            master_seed: Master seed for fixed test split
        """
        # Determine architectures
        if architectures is not None:
            self.architectures = architectures
        elif architecture is not None:
            self.architectures = [architecture]
        else:
            raise ValueError("Must specify either architecture or architectures")
        
        for arch in self.architectures:
            if arch not in AVAILABLE_ARCHITECTURES:
                raise ValueError(f"Unknown architecture: {arch}")
        
        self.n_models = n_models
        self.model_config = model_config or {}
        self.training_config = training_config or TrainingConfig()
        self.tuned_config_dir = tuned_config_dir
        self.output_dir = Path(output_dir)
        self.master_seed = master_seed
        
        # State
        self.models: List[Dict[str, Any]] = []
        self.predictions: List[np.ndarray] = []
        self.test_targets: Optional[np.ndarray] = None
        self.norm_stats: Optional[Dict] = None
        self.results: Optional[Dict] = None
    
    def run(self) -> Dict[str, Any]:
        """Run the ensemble training workflow.
        
        Returns:
            Dictionary with ensemble results and metrics
        """
        config = self.training_config
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_str = "_".join(self.architectures)
        run_dir = self.output_dir / f"ensemble_{arch_str}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Print header
        self._print_header()
        
        # Load tuned configs if specified
        tuned_configs = None
        if self.tuned_config_dir:
            tuned_configs = self._load_tuned_configs()
        
        # Train all models
        total_models = len(self.architectures) * self.n_models
        model_idx = 0
        
        for arch in self.architectures:
            for i in range(self.n_models):
                model_idx += 1
                
                # Generate seed for this model's train/val split
                train_val_seed = self.master_seed + model_idx * 1000
                
                print(f"\n{'─'*70}")
                print(f"Training Model {model_idx}/{total_models}: {arch.upper()} "
                      f"(seed={train_val_seed})")
                print(f"{'─'*70}")
                
                # Get model config (from tuned or default)
                if tuned_configs and arch in tuned_configs:
                    # Use tuned config (cycle through if fewer configs than models)
                    config_idx = i % len(tuned_configs[arch])
                    model_cfg = tuned_configs[arch][config_idx]
                else:
                    model_cfg = self.model_config
                
                # Train model
                model_results = self._train_single_model(
                    architecture=arch,
                    model_config=model_cfg,
                    train_val_seed=train_val_seed,
                    model_dir=run_dir / f"model_{model_idx:02d}_{arch}",
                )
                
                self.models.append({
                    'architecture': arch,
                    'seed': train_val_seed,
                    'model_idx': model_idx,
                    **model_results,
                })
                
                self.predictions.append(model_results['predictions'])
        
        # Evaluate ensemble
        print(f"\n{'='*70}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*70}")
        
        self.results = self._evaluate_ensemble()
        
        # Save results
        self._save_results(run_dir)
        
        # Create visualizations
        self._create_visualizations(run_dir)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _train_single_model(
        self,
        architecture: str,
        model_config: Dict,
        train_val_seed: int,
        model_dir: Path,
    ) -> Dict[str, Any]:
        """Train a single model in the ensemble."""
        config = self.training_config
        
        # Set seed
        set_seed(train_val_seed)
        
        # Load data with fixed test split
        data_loader = DataLoader(config.data_path)
        train_loader, val_loader, test_loader, X_test, y_test, norm_stats = \
            data_loader.load_with_fixed_test_split(
                master_seed=self.master_seed,
                train_val_seed=train_val_seed,
                train_ratio=0.75,
                val_ratio=0.20,
                test_ratio=0.05,
                batch_size=config.batch_size,
            )
        
        # Store test data (same for all models)
        # Note: y_test is raw (unnormalized), we normalize it for comparison
        if self.test_targets is None:
            target_mean = norm_stats['target_mean']
            target_std = norm_stats['target_std']
            # Normalize y_test to match model predictions
            self.test_targets = (y_test - target_mean) / target_std
            self.norm_stats = norm_stats
        
        # Create model
        model = create_model(architecture, model_config)
        
        # Create trainer
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(
            model=model,
            device=config.device,
            checkpoint_dir=str(model_dir / "checkpoints"),
        )
        
        # Get training hyperparameters
        lr = model_config.get("learning_rate", config.learning_rate)
        wd = model_config.get("weight_decay", config.weight_decay)
        scheduler = model_config.get("lr_scheduler", config.scheduler)
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            learning_rate=lr,
            weight_decay=wd,
            optimizer_name=config.optimizer,
            scheduler_name=scheduler,
            loss_fn=config.loss_fn,
            show_progress=False,
            save_best=True,
            verbose=0,
        )
        
        # Get predictions on test set
        predictions = self._get_predictions(model, test_loader, config.device)
        
        # Compute metrics
        target_std = norm_stats['target_std']
        test_results = trainer.test(test_loader, target_std=target_std)
        
        print(f"  Val RMSE: {trainer.best_val_loss:.6f}, "
              f"Test RMSE: {test_results['rmse_denormalized']:.2f} kg/m³")
        
        return {
            'model': model,
            'trainer': trainer,
            'history': history,
            'test_results': test_results,
            'predictions': predictions,
            'best_val_loss': trainer.best_val_loss,
        }
    
    def _get_predictions(
        self,
        model,
        test_loader,
        device: str,
    ) -> np.ndarray:
        """Get model predictions on test set."""
        device = get_device(device)
        
        if isinstance(model, torch.nn.Module):
            model.eval()
            all_preds = []
            
            with torch.no_grad():
                for features, _ in test_loader:
                    features = features.to(device)
                    preds = model(features)
                    all_preds.append(preds.cpu().numpy())
            
            return np.concatenate(all_preds).flatten()
        else:
            # LightGBM
            all_preds = []
            for features, _ in test_loader:
                preds = model.predict(features.numpy())
                all_preds.append(preds)
            return np.concatenate(all_preds).flatten()
    
    def _evaluate_ensemble(self) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        # Stack predictions: (n_models, n_samples)
        all_preds = np.stack(self.predictions, axis=0)
        
        # Ensemble predictions (mean)
        ensemble_preds = np.mean(all_preds, axis=0)
        ensemble_std = np.std(all_preds, axis=0)
        
        # Get true targets (normalized)
        targets = self.test_targets
        
        # Denormalize for metrics
        target_mean = self.norm_stats['target_mean']
        target_std = self.norm_stats['target_std']
        
        ensemble_preds_denorm = ensemble_preds * target_std + target_mean
        targets_denorm = targets * target_std + target_mean
        individual_preds_denorm = all_preds * target_std + target_mean
        
        # Ensemble metrics
        ensemble_rmse = np.sqrt(np.mean((ensemble_preds_denorm - targets_denorm) ** 2))
        ensemble_mae = np.mean(np.abs(ensemble_preds_denorm - targets_denorm))
        
        # Individual model metrics
        individual_rmses = np.sqrt(np.mean((individual_preds_denorm - targets_denorm) ** 2, axis=1))
        individual_maes = np.mean(np.abs(individual_preds_denorm - targets_denorm), axis=1)
        
        # Uncertainty analysis
        mean_uncertainty = np.mean(ensemble_std) * target_std
        
        # Also compute normalized metrics for reference
        ensemble_rmse_norm = np.sqrt(np.mean((ensemble_preds - targets) ** 2))
        ensemble_mae_norm = np.mean(np.abs(ensemble_preds - targets))
        
        results = {
            "n_models": len(self.models),
            "architectures": list(set(m['architecture'] for m in self.models)),
            # Denormalized metrics (kg/m³)
            "ensemble_rmse_denormalized": float(ensemble_rmse),
            "ensemble_mae_denormalized": float(ensemble_mae),
            "mean_individual_rmse_denormalized": float(np.mean(individual_rmses)),
            "std_individual_rmse_denormalized": float(np.std(individual_rmses)),
            "mean_individual_mae_denormalized": float(np.mean(individual_maes)),
            "best_individual_rmse_denormalized": float(np.min(individual_rmses)),
            "worst_individual_rmse_denormalized": float(np.max(individual_rmses)),
            # Normalized metrics (for internal use)
            "ensemble_rmse_normalized": float(ensemble_rmse_norm),
            "ensemble_mae_normalized": float(ensemble_mae_norm),
            # Uncertainty
            "mean_uncertainty": float(mean_uncertainty),
            "mean_std": float(mean_uncertainty),  # Alias for backward compatibility
            # Normalization stats used
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            # Individual model details
            "individual_rmses_denormalized": individual_rmses.tolist(),
            "individual_maes_denormalized": individual_maes.tolist(),
            "model_details": [
                {
                    "architecture": m['architecture'],
                    "seed": m['seed'],
                    "val_rmse_normalized": m['best_val_loss'],
                    "test_rmse_denormalized": m['test_results']['rmse_denormalized'],
                    "test_mae_denormalized": m['test_results']['mae_denormalized'],
                }
                for m in self.models
            ],
        }
        
        return results
    
    def _load_tuned_configs(self) -> Dict[str, List[Dict]]:
        """Load tuned configurations from Optuna results."""
        config_dir = Path(self.tuned_config_dir)
        
        metadata_file = config_dir / "optuna_metadata.json"
        if not metadata_file.exists():
            print(f"Warning: No metadata found in {config_dir}")
            return {}
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        arch = metadata.get("architecture", "unknown")
        configs = [c.get("params", {}) for c in metadata.get("configs", [])]
        
        print(f"Loaded {len(configs)} tuned configs for {arch}")
        
        return {arch: configs}
    
    def _save_results(self, output_dir: Path):
        """Save ensemble results."""
        # Main results
        with open(output_dir / "ensemble_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Normalization stats
        with open(output_dir / "normalization_stats.json", 'w') as f:
            json.dump({k: v.tolist() if hasattr(v, 'tolist') else v 
                      for k, v in self.norm_stats.items()}, f, indent=2)
        
        # Predictions
        np.save(output_dir / "ensemble_predictions.npy", np.stack(self.predictions))
        np.save(output_dir / "test_targets.npy", self.test_targets)
        
        print(f"\nResults saved to: {output_dir}")
    
    def _create_visualizations(self, output_dir: Path):
        """Create ensemble analysis visualizations."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("Matplotlib not available, skipping visualizations")
            return
        
        all_preds = np.stack(self.predictions)
        ensemble_preds = np.mean(all_preds, axis=0)
        ensemble_std = np.std(all_preds, axis=0)
        targets = self.test_targets
        
        target_std = self.norm_stats['target_std']
        target_mean = self.norm_stats['target_mean']
        
        # Denormalize
        ensemble_preds_d = ensemble_preds * target_std + target_mean
        ensemble_std_d = ensemble_std * target_std
        targets_d = targets * target_std + target_mean
        
        # Figure 1: Predictions vs True
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax = axes[0]
        ax.errorbar(targets_d, ensemble_preds_d, yerr=ensemble_std_d, 
                   fmt='o', alpha=0.5, markersize=4, capsize=2)
        lims = [min(targets_d.min(), ensemble_preds_d.min()), 
                max(targets_d.max(), ensemble_preds_d.max())]
        ax.plot(lims, lims, 'r--', label='Perfect prediction')
        ax.set_xlabel('True Density (kg/m³)')
        ax.set_ylabel('Predicted Density (kg/m³)')
        ax.set_title('Ensemble Predictions vs True Values')
        ax.legend()
        
        # Error distribution
        ax = axes[1]
        errors = ensemble_preds_d - targets_d
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', label='Zero error')
        ax.axvline(np.mean(errors), color='orange', linestyle='--', 
                  label=f'Mean: {np.mean(errors):.2f}')
        ax.set_xlabel('Prediction Error (kg/m³)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_analysis.png", dpi=150)
        plt.close()
        
        # Figure 2: Model comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        individual_rmses = self.results['individual_rmses_denormalized']
        model_labels = [f"{m['architecture']}_{i+1}" 
                       for i, m in enumerate(self.models)]
        
        colors = ['steelblue' if m['architecture'] == self.architectures[0] else 'coral'
                 for m in self.models]
        
        bars = ax.bar(range(len(individual_rmses)), individual_rmses, color=colors, alpha=0.7)
        ax.axhline(self.results['ensemble_rmse_denormalized'], color='green', 
                  linestyle='--', linewidth=2, label='Ensemble RMSE')
        ax.axhline(self.results['mean_individual_rmse_denormalized'], color='orange',
                  linestyle=':', linewidth=2, label='Mean Individual RMSE')
        
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.set_ylabel('Test RMSE (kg/m³)')
        ax.set_title('Individual Model vs Ensemble Performance')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=150)
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
    
    def _print_header(self):
        """Print ensemble training header."""
        print("\n" + "=" * 70)
        print("DEEP ENSEMBLE TRAINING")
        print("=" * 70)
        print(f"Architectures:    {self.architectures}")
        print(f"Models/Arch:      {self.n_models}")
        print(f"Total Models:     {len(self.architectures) * self.n_models}")
        print(f"Master Seed:      {self.master_seed}")
        print(f"Epochs:           {self.training_config.num_epochs}")
        if self.tuned_config_dir:
            print(f"Tuned Configs:    {self.tuned_config_dir}")
        print("=" * 70)
    
    def _print_summary(self):
        """Print ensemble summary."""
        if not self.results:
            return
        
        r = self.results
        
        print(f"\n{'='*70}")
        print("ENSEMBLE RESULTS")
        print(f"{'='*70}")
        print(f"Number of Models:          {r['n_models']}")
        print(f"Architectures:             {r['architectures']}")
        print(f"")
        print(f"--- Denormalized Metrics (kg/m³) ---")
        print(f"Ensemble RMSE:             {r['ensemble_rmse_denormalized']:.2f}")
        print(f"Ensemble MAE:              {r['ensemble_mae_denormalized']:.2f}")
        print(f"Mean Individual RMSE:      {r['mean_individual_rmse_denormalized']:.2f}")
        print(f"Best Individual RMSE:      {r['best_individual_rmse_denormalized']:.2f}")
        print(f"Worst Individual RMSE:     {r['worst_individual_rmse_denormalized']:.2f}")
        print(f"Mean Uncertainty (std):    {r['mean_uncertainty']:.2f}")
        print(f"")
        print(f"--- Normalized Metrics ---")
        print(f"Ensemble RMSE:             {r['ensemble_rmse_normalized']:.6f}")
        print(f"Ensemble MAE:              {r['ensemble_mae_normalized']:.6f}")
        print(f"")
        print(f"--- Normalization Stats ---")
        print(f"Target Mean:               {r['target_mean']:.2f} kg/m³")
        print(f"Target Std:                {r['target_std']:.2f} kg/m³")
        
        improvement = (r['mean_individual_rmse_denormalized'] - 
                      r['ensemble_rmse_denormalized'])
        if improvement > 0:
            pct = improvement / r['mean_individual_rmse_denormalized'] * 100
            print(f"")
            print(f"Ensemble Improvement:      {improvement:.2f} kg/m³ ({pct:.1f}%)")
        
        print("=" * 70 + "\n")


__all__ = ["EnsembleTrainer"]
