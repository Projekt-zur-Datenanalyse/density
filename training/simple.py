"""
Simple single-model training workflow.

This module provides a clean, high-level interface for training
a single model with progress tracking, evaluation, and results saving.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from core import TrainingConfig, DataLoader, Trainer, set_seed, get_device
from models import create_model, get_model_info, AVAILABLE_ARCHITECTURES


class SimpleTrainer:
    """High-level trainer for single model training.
    
    Provides a simple interface for:
    - Loading data with proper splits
    - Training with progress tracking
    - Evaluation on test set
    - Saving results and checkpoints
    
    Example:
        trainer = SimpleTrainer(
            architecture="mlp",
            model_config={"hidden_dims": [256, 64, 32]},
            training_config=TrainingConfig(num_epochs=100),
        )
        results = trainer.run()
    """
    
    def __init__(
        self,
        architecture: str,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[TrainingConfig] = None,
        tuned_config_dir: Optional[str] = None,
    ):
        """Initialize the trainer.
        
        Args:
            architecture: Model architecture (mlp, cnn, cnn_multiscale, lightgbm)
            model_config: Model-specific configuration. Overridden by tuned_config if provided.
            training_config: Training configuration. If None, uses defaults.
            tuned_config_dir: Path to Optuna results directory to load tuned hyperparameters.
                            If provided, model_config is ignored.
        """
        if architecture not in AVAILABLE_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Available: {AVAILABLE_ARCHITECTURES}"
            )
        
        self.architecture = architecture
        self.training_config = training_config or TrainingConfig()
        
        # Load tuned config if specified
        if tuned_config_dir:
            self.model_config = self._load_tuned_config(tuned_config_dir)
            self.using_tuned = True
        else:
            self.model_config = model_config or {}
            self.using_tuned = False
        
        # State
        self.model = None
        self.trainer = None
        self.data_stats = None
        self.results = None
    
    def _load_tuned_config(self, config_dir: str) -> Dict[str, Any]:
        """Load tuned hyperparameters from Optuna results."""
        config_path = Path(config_dir)
        
        # Try loading from metadata file
        metadata_file = config_path / "optuna_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            if metadata.get("configs"):
                # Get best config (rank 1)
                best_config = metadata["configs"][0]
                print(f"Loaded tuned config from {config_dir}")
                print(f"  Trial {best_config['trial_number']}, "
                      f"Validation RMSE: {best_config['validation_value']:.6f}")
                return best_config.get("params", {})
        
        # Try loading from individual config file
        config_file = config_path / "configs" / "config_rank_01.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            print(f"Loaded tuned config from {config_file}")
            return config.get("params", {})
        
        raise FileNotFoundError(f"No tuned config found in {config_dir}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete training workflow.
        
        Returns:
            Dictionary with training results and metrics
        """
        config = self.training_config
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Get device
        device = get_device(config.device)
        
        # Print header
        self._print_header()
        
        # Load data
        print("\n[1/4] Loading data...")
        data_loader = DataLoader(config.data_path)
        train_loader, val_loader, test_loader, stats = data_loader.load(
            validation_split=config.validation_split,
            test_split=config.test_split,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )
        self.data_stats = stats
        
        print(f"      Train: {stats['train_samples']} samples")
        print(f"      Val:   {stats['val_samples']} samples")
        print(f"      Test:  {stats['test_samples']} samples")
        
        # Create model
        print("\n[2/4] Creating model...")
        self.model = create_model(self.architecture, self.model_config)
        print(get_model_info(self.model))
        
        # Create trainer
        output_dir = Path(config.output_dir) / f"{self.architecture}_{datetime.now():%Y%m%d_%H%M%S}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = str(output_dir / "checkpoints") if config.save_checkpoints else None
        self.trainer = Trainer(
            model=self.model,
            device=str(device),
            checkpoint_dir=checkpoint_dir,
        )
        
        # Train
        print("\n[3/4] Training...")
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            learning_rate=self._get_learning_rate(),
            weight_decay=self._get_weight_decay(),
            optimizer_name=config.optimizer,
            scheduler_name=config.scheduler,
            loss_fn=config.loss_fn,
            show_progress=config.show_progress_bar,
            save_best=config.save_checkpoints,
            verbose=config.verbose,
        )
        
        # Test
        print("\n[4/4] Evaluating on test set...")
        target_std = stats['normalization']['target_std']
        test_results = self.trainer.test(test_loader, target_std=target_std)
        
        # Compile results
        self.results = {
            "architecture": self.architecture,
            "using_tuned_config": self.using_tuned,
            "model_config": self.model_config,
            "training_config": config.to_dict(),
            "data_stats": stats,
            "training_history": history,
            "test_results": test_results,
            "best_val_loss": self.trainer.best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results
        self._save_results(output_dir)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _get_learning_rate(self) -> float:
        """Get learning rate from tuned config or training config."""
        if self.using_tuned and "learning_rate" in self.model_config:
            return float(self.model_config["learning_rate"])
        return self.training_config.learning_rate
    
    def _get_weight_decay(self) -> float:
        """Get weight decay from tuned config or training config."""
        if self.using_tuned and "weight_decay" in self.model_config:
            return float(self.model_config["weight_decay"])
        return self.training_config.weight_decay
    
    def _print_header(self):
        """Print training header."""
        config = self.training_config
        print("\n" + "=" * 70)
        print("CHEMICAL DENSITY SURROGATE - TRAINING")
        print("=" * 70)
        print(f"Architecture:  {self.architecture.upper()}")
        print(f"Tuned Config:  {'Yes' if self.using_tuned else 'No'}")
        print(f"Epochs:        {config.num_epochs}")
        print(f"Batch Size:    {config.batch_size}")
        print(f"Learning Rate: {self._get_learning_rate()}")
        print(f"Optimizer:     {config.optimizer}")
        print(f"Scheduler:     {config.scheduler}")
        print(f"Seed:          {config.seed}")
        print("=" * 70)
    
    def _print_summary(self):
        """Print final summary."""
        if not self.results:
            return
        
        test = self.results['test_results']
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Validation RMSE: {self.results['best_val_loss']:.6f}")
        print(f"Test RMSE (normalized): {test['rmse_normalized']:.6f}")
        if 'rmse_denormalized' in test:
            print(f"Test RMSE (kg/m³):      {test['rmse_denormalized']:.2f}")
            print(f"Test MAE  (kg/m³):      {test['mae_denormalized']:.2f}")
        print("=" * 70 + "\n")
    
    def _save_results(self, output_dir: Path):
        """Save results to disk."""
        # Save complete results
        with open(output_dir / "results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = self._prepare_for_json(self.results)
            json.dump(results_json, f, indent=2)
        
        # Save normalization stats separately (useful for inference)
        with open(output_dir / "normalization_stats.json", 'w') as f:
            json.dump(self.data_stats['normalization'], f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    
    def _prepare_for_json(self, obj):
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        return obj


def find_latest_tuned_config(architecture: str) -> Optional[str]:
    """Find the latest Optuna results directory for an architecture.
    
    Args:
        architecture: Model architecture name
        
    Returns:
        Path to latest results directory, or None if not found
    """
    import glob
    
    # Look for optuna results directories
    pattern = f"optuna_results_{architecture}_*"
    dirs = sorted(glob.glob(pattern), reverse=True)
    
    if dirs:
        return dirs[0]
    
    # Also check in tuning/ subdirectory
    pattern = f"tuning/results/{architecture}_*"
    dirs = sorted(glob.glob(pattern), reverse=True)
    
    return dirs[0] if dirs else None


__all__ = ["SimpleTrainer", "find_latest_tuned_config"]
