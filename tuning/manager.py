"""
Hyperparameter tuning manager using Optuna.

This module provides the main interface for running hyperparameter
optimization and managing results.
"""

import json
import optuna
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .search_spaces import SearchSpace, SEARCH_CONFIGS
from .objective import create_objective
from core.utils import get_device


class TuningResults:
    """Manages saving and loading of tuning results."""
    
    def __init__(self, results_dir: Path):
        """Initialize results manager.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs_dir = self.results_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.results_dir / "optuna_metadata.json"
    
    def save_n_best(
        self,
        study: optuna.study.Study,
        architecture: str,
        n_best: int = 5,
        max_epochs: int = 100,
        seed: int = 46,
    ) -> Dict[str, Any]:
        """Save n-best hyperparameter configurations.
        
        Args:
            study: Optuna study object
            architecture: Model architecture
            n_best: Number of best configs to save
            max_epochs: Maximum epochs used in tuning
            seed: Master seed used
            
        Returns:
            Metadata dictionary
        """
        # Get completed trials sorted by value
        completed_trials = [t for t in study.trials if t.value is not None]
        completed_trials.sort(key=lambda t: t.value)
        
        n_best = min(n_best, len(completed_trials))
        best_trials = completed_trials[:n_best]
        
        metadata = {
            "architecture": architecture,
            "n_best": n_best,
            "timestamp": datetime.now().isoformat(),
            "n_total_trials": len(study.trials),
            "max_epochs": max_epochs,
            "master_seed": seed,
            "configs": [],
        }
        
        for rank, trial in enumerate(best_trials, 1):
            config_data = {
                "rank": rank,
                "trial_number": trial.number,
                "validation_value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "architecture": architecture,
                "max_epochs": max_epochs,
                "seed": seed + rank,
            }
            
            metadata["configs"].append(config_data)
            
            # Save individual config file
            config_file = self.configs_dir / f"config_rank_{rank:02d}.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def load_n_best(self, n_best: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load n-best configurations.
        
        Args:
            n_best: Number of configs to load (None = all)
            
        Returns:
            List of config dictionaries
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_file}")
        
        with open(self.metadata_file) as f:
            metadata = json.load(f)
        
        configs = metadata["configs"]
        if n_best is not None:
            configs = configs[:n_best]
        
        return configs
    
    def get_best(self) -> Dict[str, Any]:
        """Get the best configuration."""
        configs = self.load_n_best(n_best=1)
        return configs[0] if configs else None


class HyperparameterTuner:
    """Main interface for hyperparameter tuning with Optuna."""
    
    def __init__(
        self,
        architecture: str,
        n_trials: int = 50,
        max_epochs: int = 100,
        n_best_save: int = 5,
        data_path: str = "dataset.csv",
        device: str = "auto",
        seed: int = 46,
        output_dir: Optional[str] = None,
    ):
        """Initialize the tuner.
        
        Args:
            architecture: Model architecture to tune
            n_trials: Number of optimization trials
            max_epochs: Maximum epochs per trial
            n_best_save: Number of best configs to save
            data_path: Path to dataset
            device: Device to use
            seed: Master seed for reproducibility
            output_dir: Directory for results (auto-generated if None)
        """
        self.architecture = architecture
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.n_best_save = n_best_save
        self.data_path = data_path
        self.device = str(get_device(device))
        self.seed = seed
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results_tuning/{architecture}_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results handler
        self.results = TuningResults(self.output_dir)
        
        # Study configuration
        self.study_name = f"tune_{architecture}_{datetime.now():%Y%m%d_%H%M%S}"
        self.storage_url = f"sqlite:///{self.output_dir}/study.db"
    
    def run(
        self,
        sampler: str = "tpe",
        pruner: str = "median",
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> Tuple[optuna.study.Study, Dict[str, Any]]:
        """Run hyperparameter optimization.
        
        Args:
            sampler: Sampler type ("tpe", "random", or "grid")
            pruner: Pruner type ("median", "noop", or "percentile")
            n_jobs: Number of parallel jobs
            verbose: Show progress
            
        Returns:
            Tuple of (Study object, results metadata)
        """
        if verbose:
            self._print_header()
        
        # Create sampler
        if sampler == "tpe":
            optuna_sampler = optuna.samplers.TPESampler(seed=self.seed)
        elif sampler == "random":
            optuna_sampler = optuna.samplers.RandomSampler(seed=self.seed)
        else:
            optuna_sampler = optuna.samplers.TPESampler(seed=self.seed)
        
        # Create pruner
        if pruner == "median":
            optuna_pruner = optuna.pruners.MedianPruner()
        elif pruner == "noop":
            optuna_pruner = optuna.pruners.NopPruner()
        elif pruner == "percentile":
            optuna_pruner = optuna.pruners.PercentilePruner(percentile=30.0)
        else:
            optuna_pruner = optuna.pruners.MedianPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction="minimize",
            sampler=optuna_sampler,
            pruner=optuna_pruner,
            load_if_exists=False,
        )
        
        # Get objective function
        suggest_fn = SearchSpace.get_suggest_function(self.architecture)
        objective = create_objective(
            architecture=self.architecture,
            num_epochs=self.max_epochs,
            data_path=self.data_path,
            device=self.device,
            suggest_fn=suggest_fn,
            seed=self.seed,
        )
        
        # Run optimization
        if verbose:
            print(f"Starting optimization with {self.n_trials} trials...")
            print(f"Sampler: {sampler}, Pruner: {pruner}\n")
        
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=n_jobs,
                show_progress_bar=verbose,
            )
        except KeyboardInterrupt:
            if verbose:
                print("\nOptimization interrupted by user")
        
        # Save results
        metadata = self.results.save_n_best(
            study=study,
            architecture=self.architecture,
            n_best=self.n_best_save,
            max_epochs=self.max_epochs,
            seed=self.seed,
        )
        
        if verbose:
            self._print_summary(study)
            print(f"\nResults saved to: {self.output_dir}")
        
        return study, metadata
    
    def _print_header(self):
        """Print tuning header."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("=" * 70)
        print(f"Architecture:  {self.architecture.upper()}")
        print(f"Trials:        {self.n_trials}")
        print(f"Max Epochs:    {self.max_epochs}")
        print(f"N Best Save:   {self.n_best_save}")
        print(f"Device:        {self.device.upper()}")
        print(f"Seed:          {self.seed}")
        print(f"Output:        {self.output_dir}")
        print("=" * 70 + "\n")
    
    def _print_summary(self, study: optuna.study.Study):
        """Print optimization summary."""
        best = study.best_trial
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total Trials:      {len(study.trials)}")
        print(f"Best Trial:        {best.number}")
        print(f"")
        print(f"--- Validation Metric (Normalized) ---")
        print(f"Val RMSE:          {best.value:.6f}")
        print(f"")
        print(f"--- Test Metrics (Denormalized, kg/m³) ---")
        if 'test_rmse_denorm' in best.user_attrs:
            print(f"Test RMSE:         {best.user_attrs['test_rmse_denorm']:.2f}")
        if 'test_mae_denorm' in best.user_attrs:
            print(f"Test MAE:          {best.user_attrs['test_mae_denorm']:.2f}")
        
        print(f"")
        print(f"Best Hyperparameters:")
        for key, value in sorted(best.params.items()):
            print(f"  {key}: {value}")
        
        print("=" * 70)


def find_latest_results(architecture: str) -> Optional[Path]:
    """Find the latest tuning results for an architecture.
    
    Args:
        architecture: Model architecture
        
    Returns:
        Path to latest results directory, or None
    """
    import glob
    
    patterns = [
        f"results_tuning/{architecture}_*",
        f"optuna_results_{architecture}_*",
    ]
    
    for pattern in patterns:
        dirs = sorted(glob.glob(pattern), reverse=True)
        if dirs:
            return Path(dirs[0])
    
    return None


__all__ = ["HyperparameterTuner", "TuningResults", "find_latest_results"]
