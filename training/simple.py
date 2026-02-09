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
        config = TrainingConfig(
            architecture="mlp",
            model_config={"hidden_dims": [16, 32, 8], "activation": "relu"},
        )
        trainer = SimpleTrainer(config)
        results = trainer.run()
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        verbose: bool = True,
    ):
        """Initialize the trainer.
        
        Args:
            config: Training configuration including architecture and model settings
            verbose: Whether to print progress messages
        """
        if config.architecture not in AVAILABLE_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{config.architecture}'. "
                f"Available: {AVAILABLE_ARCHITECTURES}"
            )
        
        self.config = config
        self.verbose = verbose
        
        # State
        self.model = None
        self.trainer = None
        self.data_stats = None
        self.results = None
        self.output_dir = None
        self._using_tuned = False
    
    def load_tuned_config(self, config_dir: str, rank: int = 1) -> Dict[str, Any]:
        """Load tuned hyperparameters from Optuna results.
        
        Args:
            config_dir: Path to Optuna results directory
            rank: Rank of config to load (1 = best)
            
        Returns:
            Loaded hyperparameters dict
        """
        config_path = Path(config_dir)
        
        # Try loading from metadata file
        metadata_file = config_path / "optuna_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            if metadata.get("configs") and len(metadata["configs"]) >= rank:
                tuned = metadata["configs"][rank - 1]
                if self.verbose:
                    print(f"Loaded tuned config rank {rank} from {config_dir}")
                    print(f"  Trial {tuned['trial_number']}, "
                          f"Validation RMSE: {tuned['validation_value']:.6f}")
                
                # Merge tuned params into model_config
                params = tuned.get("params", {})
                self.config.model_config.update(params)
                
                # Update training params if present
                if "learning_rate" in params:
                    self.config.learning_rate = float(params["learning_rate"])
                if "weight_decay" in params:
                    self.config.weight_decay = float(params["weight_decay"])
                if "batch_size" in params:
                    self.config.batch_size = int(params["batch_size"])
                
                self._using_tuned = True
                return params
        
        # Try loading from individual config file
        config_file = config_path / "configs" / f"config_rank_{rank:02d}.json"
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
            if self.verbose:
                print(f"Loaded tuned config from {config_file}")
            
            params = config_data.get("params", {})
            self.config.model_config.update(params)
            self._using_tuned = True
            return params
        
        raise FileNotFoundError(f"No tuned config found in {config_dir}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete training workflow.
        
        Returns:
            Dictionary with training results and metrics
        """
        config = self.config
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Get device
        device = get_device(config.device)
        
        # Print header
        if self.verbose:
            self._print_header()
        
        # Load data
        if self.verbose:
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
        
        if self.verbose:
            print(f"      Train: {stats['train_samples']} samples")
            print(f"      Val:   {stats['val_samples']} samples")
            print(f"      Test:  {stats['test_samples']} samples")
        
        # Create model
        if self.verbose:
            print("\n[2/4] Creating model...")
        self.model = create_model(config.architecture, config.model_config)
        if self.verbose:
            print(f"      {self.model}")
        
        # Create output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
        else:
            self.output_dir = Path(f"results_training/{config.architecture}_{datetime.now():%Y%m%d_%H%M%S}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = str(self.output_dir / "checkpoints") if config.save_checkpoints else None
        self.trainer = Trainer(
            model=self.model,
            device=str(device),
            checkpoint_dir=checkpoint_dir,
        )
        
        # Train
        if self.verbose:
            print("\n[3/4] Training...")
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            optimizer_name=config.optimizer,
            scheduler_name=config.scheduler,
            loss_fn=config.loss_fn,
            show_progress=config.show_progress_bar,
            save_best=config.save_checkpoints,
            verbose=config.verbose,
        )
        
        # Test (with predictions for saving)
        if self.verbose:
            print("\n[4/4] Evaluating on test set...")
        target_std = stats['normalization']['target_std']
        test_results = self.trainer.test(
            test_loader, 
            target_std=target_std, 
            return_predictions=True
        )
        
        # Extract predictions for separate saving
        predictions = test_results.pop("predictions", None)
        targets = test_results.pop("targets", None)
        
        # Compile results
        self.results = {
            "architecture": config.architecture,
            "using_tuned_config": self._using_tuned,
            "model_config": config.model_config,
            "training_config": config.to_dict(),
            "data_stats": stats,
            "training_history": history,
            "test_results": test_results,
            "test_rmse_denormalized": test_results.get("rmse_denormalized", 
                                                        test_results["rmse_normalized"] * target_std),
            "best_val_loss": self.trainer.best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Store predictions for saving
        self._predictions = predictions
        self._targets = targets
        
        # Save results
        self._save_results(self.output_dir)
        
        # Print summary
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def _print_header(self):
        """Print training header."""
        config = self.config
        print("\n" + "=" * 70)
        print("CHEMICAL DENSITY SURROGATE - TRAINING")
        print("=" * 70)
        print(f"Architecture:  {config.architecture.upper()}")
        if config.architecture == "mlp":
            dims = config.model_config.get("hidden_dims", [16, 32, 8])
            act = config.model_config.get("activation", "relu")
            print(f"Hidden Dims:   {dims}")
            print(f"Activation:    {act}")
        print(f"Tuned Config:  {'Yes' if self._using_tuned else 'No'}")
        print(f"Epochs:        {config.num_epochs}")
        print(f"Batch Size:    {config.batch_size}")
        print(f"Learning Rate: {config.learning_rate}")
        print(f"Optimizer:     {config.optimizer}")
        print(f"Scheduler:     {config.scheduler}")
        print(f"Seed:          {config.seed}")
        print("=" * 70)
    
    def _print_summary(self):
        """Print final summary."""
        if not self.results:
            return
        
        test = self.results['test_results']
        stats = self.data_stats
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Validation RMSE (normalized): {self.results['best_val_loss']:.6f}")
        print(f"")
        print(f"--- Denormalized Test Metrics (kg/m³) ---")
        if 'rmse_denormalized' in test:
            print(f"Test RMSE:                         {test['rmse_denormalized']:.2f}")
            print(f"Test MAE:                          {test['mae_denormalized']:.2f}")
        print(f"")
        print(f"--- Normalized Test Metrics ---")
        print(f"Test RMSE:                         {test['rmse_normalized']:.6f}")
        print(f"Test MAE:                          {test['mae_normalized']:.6f}")
        print(f"")
        print(f"--- Normalization Stats ---")
        print(f"Target Mean:                       {stats['normalization']['target_mean']:.2f} kg/m³")
        print(f"Target Std:                        {stats['normalization']['target_std']:.2f} kg/m³")
        print("=" * 70 + "\n")
    
    def _save_results(self, output_dir: Path):
        """Save results to disk."""
        # Save complete results
        with open(output_dir / "results.json", 'w') as f:
            results_json = self._prepare_for_json(self.results)
            json.dump(results_json, f, indent=2)
        
        # Save normalization stats separately (useful for inference)
        with open(output_dir / "normalization_stats.json", 'w') as f:
            json.dump(self._prepare_for_json(self.data_stats['normalization']), f, indent=2)
        
        # Save model config
        with open(output_dir / "model_config.json", 'w') as f:
            json.dump(self._prepare_for_json(self.config.model_config), f, indent=2)
        
        # Save test results
        with open(output_dir / "test_results.json", 'w') as f:
            json.dump(self._prepare_for_json(self.results['test_results']), f, indent=2)
        
        # Save training history separately (for plotting)
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(self._prepare_for_json(self.results['training_history']), f, indent=2)
        
        # Save predictions (for analysis/plotting)
        if self._predictions is not None and self._targets is not None:
            torch.save({
                "predictions": torch.from_numpy(self._predictions),
                "targets": torch.from_numpy(self._targets),
            }, output_dir / "predictions.pt")
        
        if self.verbose:
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


__all__ = ["SimpleTrainer"]
