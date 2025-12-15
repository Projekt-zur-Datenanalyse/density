"""
Main training script for the Chemical Density Surrogate Model.

This script demonstrates how to:
1. Load and prepare data
2. Create model with custom configuration
3. Train the model
4. Evaluate on test set
"""

import torch
import argparse
from pathlib import Path
import json
import sys
import numpy as np

from config import ModelConfig, TrainingConfig
from model import ChemicalDensitySurrogate
from data_loader import ChemicalDensityDataLoader
from trainer import ModelTrainer

# Import optional architectures
try:
    from cnn_model import ConvolutionalSurrogate, MultiScaleConvolutionalSurrogate
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    from lightgbm_model import LightGBMSurrogate
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments with defaults from config.py."""
    # Load default configs
    default_model_config = ModelConfig()
    default_training_config = TrainingConfig()
    
    parser = argparse.ArgumentParser(
        description="Train Chemical Density Surrogate Model"
    )
    
    # Architecture selection
    parser.add_argument(
        "--architecture",
        type=str,
        default=default_model_config.architecture,
        choices=["mlp", "cnn", "cnn_multiscale", "lightgbm"],
        help=f"Model architecture to train (default: {default_model_config.architecture})",
    )
    
    # MLP configuration
    parser.add_argument(
        "--num-layers",
        type=int,
        default=default_model_config.num_layers,
        help=f"Number of hidden layers in MLP (default: {default_model_config.num_layers}). Ignored if --hidden-layer-dims is specified.",
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=int(default_model_config.expansion_factor),
        help=f"Hidden dimension multiplier for MLP (default: {int(default_model_config.expansion_factor)}). Ignored if --hidden-layer-dims is specified.",
    )
    parser.add_argument(
        "--hidden-layer-dims",
        type=int,
        nargs='*',
        default=None,
        help="Custom hidden layer dimensions for MLP (space-separated integers). E.g., --hidden-layer-dims 32 16 8. Empty list (--hidden-layer-dims without values) creates direct input->output projection.",
    )
    parser.add_argument(
        "--use-swiglu",
        action="store_true",
        default=default_model_config.use_swiglu,
        help="Use SwiGLU activation for MLP",
    )
    
    # Common regularization
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=default_model_config.dropout_rate,
        help=f"Dropout rate for all architectures (default: {default_model_config.dropout_rate})",
    )
    
    # CNN configuration
    parser.add_argument(
        "--cnn-expansion-size",
        type=int,
        default=default_model_config.cnn_expansion_size,
        help=f"Expand input features to NxN spatial grid (default: {default_model_config.cnn_expansion_size}x{default_model_config.cnn_expansion_size})",
    )
    parser.add_argument(
        "--cnn-num-layers",
        type=int,
        default=default_model_config.cnn_num_layers,
        help=f"Number of convolutional layers (default: {default_model_config.cnn_num_layers})",
    )
    parser.add_argument(
        "--cnn-kernel-size",
        type=int,
        default=default_model_config.cnn_kernel_size,
        help=f"Convolutional kernel size (default: {default_model_config.cnn_kernel_size})",
    )
    parser.add_argument(
        "--cnn-no-batch-norm",
        action="store_true",
        default=not default_model_config.cnn_use_batch_norm,
        help="Disable batch normalization in CNN",
    )
    parser.add_argument(
        "--cnn-no-residual",
        action="store_true",
        default=not default_model_config.cnn_use_residual,
        help="Disable residual connections in CNN",
    )
    
    # Multi-Scale CNN configuration
    parser.add_argument(
        "--cnn-ms-expansion-size",
        type=int,
        default=default_model_config.cnn_multiscale_expansion_size,
        help=f"Spatial expansion size for Multi-Scale CNN (default: {default_model_config.cnn_multiscale_expansion_size})",
    )
    parser.add_argument(
        "--cnn-ms-num-scales",
        type=int,
        default=default_model_config.cnn_multiscale_num_scales,
        help=f"Number of parallel scales in Multi-Scale CNN (default: {default_model_config.cnn_multiscale_num_scales})",
    )
    parser.add_argument(
        "--cnn-ms-base-channels",
        type=int,
        default=default_model_config.cnn_multiscale_base_channels,
        help=f"Base channels per scale in Multi-Scale CNN (default: {default_model_config.cnn_multiscale_base_channels})",
    )
    
    # LightGBM configuration
    parser.add_argument(
        "--lgb-num-leaves",
        type=int,
        default=default_model_config.lgb_num_leaves,
        help=f"LightGBM - max leaves per tree (default: {default_model_config.lgb_num_leaves})",
    )
    parser.add_argument(
        "--lgb-learning-rate",
        type=float,
        default=default_model_config.lgb_learning_rate,
        help=f"LightGBM - learning rate (default: {default_model_config.lgb_learning_rate})",
    )
    parser.add_argument(
        "--lgb-num-boost-round",
        type=int,
        default=default_model_config.lgb_num_boost_round,
        help=f"LightGBM - number of boosting rounds (default: {default_model_config.lgb_num_boost_round})",
    )
    parser.add_argument(
        "--lgb-max-depth",
        type=int,
        default=default_model_config.lgb_max_depth,
        help=f"LightGBM - max tree depth (default: {default_model_config.lgb_max_depth})",
    )
    parser.add_argument(
        "--lgb-min-child-samples",
        type=int,
        default=default_model_config.lgb_min_child_samples,
        help=f"LightGBM - min samples in leaf (default: {default_model_config.lgb_min_child_samples})",
    )
    parser.add_argument(
        "--lgb-subsample",
        type=float,
        default=default_model_config.lgb_subsample,
        help=f"LightGBM - subsample ratio (default: {default_model_config.lgb_subsample})",
    )
    parser.add_argument(
        "--lgb-colsample-bytree",
        type=float,
        default=default_model_config.lgb_colsample_bytree,
        help=f"LightGBM - feature subsample ratio (default: {default_model_config.lgb_colsample_bytree})",
    )
    parser.add_argument(
        "--lgb-reg-alpha",
        type=float,
        default=default_model_config.lgb_reg_alpha,
        help=f"LightGBM - L1 regularization (default: {default_model_config.lgb_reg_alpha})",
    )
    parser.add_argument(
        "--lgb-reg-lambda",
        type=float,
        default=default_model_config.lgb_reg_lambda,
        help=f"LightGBM - L2 regularization (default: {default_model_config.lgb_reg_lambda})",
    )
    parser.add_argument(
        "--lgb-boosting-type",
        type=str,
        default=default_model_config.lgb_boosting_type,
        choices=["gbdt", "rf", "dart", "goss"],
        help=f"LightGBM - boosting type (default: {default_model_config.lgb_boosting_type})",
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_training_config.num_epochs,
        help=f"Number of training epochs (default: {default_training_config.num_epochs})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_training_config.batch_size,
        help=f"Batch size for training (default: {default_training_config.batch_size})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_training_config.learning_rate,
        help=f"Learning rate (default: {default_training_config.learning_rate})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=default_training_config.optimizer,
        choices=["adam", "sgd"],
        help=f"Optimizer to use (default: {default_training_config.optimizer})",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=default_training_config.loss_fn,
        choices=["mse", "mae"],
        help=f"Loss function (default: {default_training_config.loss_fn})",
    )
    
    # Learning rate scheduler configuration
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=default_training_config.lr_scheduler,
        choices=["none", "onecycle", "cosine"],
        help=f"Learning rate scheduler (default: {default_training_config.lr_scheduler})",
    )
    parser.add_argument(
        "--onecycle-pct-start",
        type=float,
        default=default_training_config.onecycle_pct_start,
        help=f"OneCycleLR - percent of cycle spent increasing LR (default: {default_training_config.onecycle_pct_start})",
    )
    parser.add_argument(
        "--cosine-t-max",
        type=int,
        default=default_training_config.cosine_t_max,
        help=f"CosineAnnealingLR - max iterations (default: {default_training_config.cosine_t_max})",
    )
    parser.add_argument(
        "--cosine-eta-min",
        type=float,
        default=default_training_config.cosine_eta_min,
        help=f"CosineAnnealingLR - minimum learning rate (default: {default_training_config.cosine_eta_min})",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        default=True,
        help="Show progress bar for each batch (default: True)",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Hide progress bar for each batch",
    )
    
    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing dataset CSV files (default: current directory)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=default_training_config.validation_split,
        help=f"Validation split ratio (default: {default_training_config.validation_split})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=default_training_config.test_split,
        help=f"Test split ratio (default: {default_training_config.test_split})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=default_training_config.weight_decay,
        help=f"L2 regularization weight decay (default: {default_training_config.weight_decay})",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable input/target normalization",
    )
    
    # General
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_training_config.random_seed,
        help=f"Random seed for reproducibility (default: {default_training_config.random_seed})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    
    return parser.parse_args()


def create_model(args, model_config: ModelConfig):
    """Create model based on architecture specified in config."""
    architecture = model_config.architecture
    
    print(f"[Architecture] Creating {architecture.upper()} model...")
    
    if architecture == "mlp":
        model = ChemicalDensitySurrogate(model_config)
    
    elif architecture == "cnn":
        if not CNN_AVAILABLE:
            print("ERROR: CNN requires cnn_model.py to be available")
            sys.exit(1)
        model = ConvolutionalSurrogate(
            num_input_features=model_config.input_dim,
            expansion_size=model_config.cnn_expansion_size,
            num_conv_layers=model_config.cnn_num_layers,
            kernel_size=model_config.cnn_kernel_size,
            use_batch_norm=model_config.cnn_use_batch_norm,
            use_residual=model_config.cnn_use_residual,
            dropout_rate=model_config.dropout_rate,
        )
    
    elif architecture == "cnn_multiscale":
        if not CNN_AVAILABLE:
            print("ERROR: CNN Multi-Scale requires cnn_model.py to be available")
            sys.exit(1)
        model = MultiScaleConvolutionalSurrogate(
            num_input_features=model_config.input_dim,
            expansion_size=model_config.cnn_multiscale_expansion_size,
            num_scales=model_config.cnn_multiscale_num_scales,
            base_channels=model_config.cnn_multiscale_base_channels,
            dropout_rate=model_config.dropout_rate,
        )
    
    elif architecture == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            print("ERROR: LightGBM requires lightgbm package. Install with: pip install lightgbm")
            sys.exit(1)
        model = LightGBMSurrogate(
            num_leaves=model_config.lgb_num_leaves,
            learning_rate=model_config.lgb_learning_rate,
            num_boost_round=model_config.lgb_num_boost_round,
            max_depth=model_config.lgb_max_depth,
            min_child_samples=model_config.lgb_min_child_samples,
            subsample=model_config.lgb_subsample,
            colsample_bytree=model_config.lgb_colsample_bytree,
            reg_alpha=model_config.lgb_reg_alpha,
            reg_lambda=model_config.lgb_reg_lambda,
            boosting_type=model_config.lgb_boosting_type,
            metric=model_config.lgb_metric,
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if hasattr(model, 'get_model_info'):
        print(model.get_model_info())
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params:,}")
    
    return model


def main():
    """Main training script."""
    args = parse_arguments()
    
    # Set random seed for reproducibility across all libraries
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU
    # Ensure deterministic behavior (may slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CHEMICAL DENSITY SURROGATE MODEL TRAINING")
    print("=" * 80)
    
    # ===== CONFIGURATION =====
    print("\n[1] Setting up configuration...")
    
    # Model configuration
    model_config = ModelConfig(
        architecture=args.architecture,
        input_dim=4,
        output_dim=1,
        # MLP settings
        num_layers=args.num_layers,
        expansion_factor=args.expansion_factor,
        hidden_layer_dims=args.hidden_layer_dims,  # Use custom layer dims if provided
        use_swiglu=args.use_swiglu,
        # CNN settings
        cnn_expansion_size=args.cnn_expansion_size,
        cnn_num_layers=args.cnn_num_layers,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_use_batch_norm=not args.cnn_no_batch_norm,
        cnn_use_residual=not args.cnn_no_residual,
        # Multi-Scale CNN settings
        cnn_multiscale_expansion_size=args.cnn_ms_expansion_size,
        cnn_multiscale_num_scales=args.cnn_ms_num_scales,
        cnn_multiscale_base_channels=args.cnn_ms_base_channels,
        # LightGBM settings
        lgb_num_leaves=args.lgb_num_leaves,
        lgb_learning_rate=args.lgb_learning_rate,
        lgb_num_boost_round=args.lgb_num_boost_round,
        lgb_max_depth=args.lgb_max_depth,
        lgb_min_child_samples=args.lgb_min_child_samples,
        lgb_subsample=args.lgb_subsample,
        lgb_colsample_bytree=args.lgb_colsample_bytree,
        lgb_reg_alpha=args.lgb_reg_alpha,
        lgb_reg_lambda=args.lgb_reg_lambda,
        lgb_boosting_type=args.lgb_boosting_type,
        # Common settings
        dropout_rate=args.dropout_rate if hasattr(args, 'dropout_rate') else 0.2,
        device=args.device,
    )
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        validation_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
        loss_fn=args.loss,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        normalize_inputs=not args.no_normalize,
        normalize_outputs=not args.no_normalize,
        checkpoint_dir=str(output_dir / "checkpoints"),
        lr_scheduler=args.lr_scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        show_progress_bar=not args.no_progress_bar,
    )
    
    # ===== MODEL CREATION =====
    print("\n[2] Creating model...")
    model = create_model(args, model_config)
    
    # ===== DATA LOADING =====
    print("\n[3] Loading data...")
    data_loader = ChemicalDensityDataLoader(args.data_dir)
    
    train_loader, val_loader, test_loader, dataset = data_loader.load_dataset(
        normalize_features=training_config.normalize_inputs,
        normalize_targets=training_config.normalize_outputs,
        validation_split=training_config.validation_split,
        test_split=training_config.test_split,
        batch_size=training_config.batch_size,
        seed=args.seed,
    )
    
    # Store normalization stats
    norm_stats = {
        'feature_mean': dataset.feature_mean.tolist() if dataset.feature_mean is not None else None,
        'feature_std': dataset.feature_std.tolist() if dataset.feature_std is not None else None,
        'target_mean': float(dataset.target_mean) if dataset.target_mean is not None else None,
        'target_std': float(dataset.target_std) if dataset.target_std is not None else None,
    }
    
    # ===== TRAINING =====
    print("\n[4] Starting training...")
    trainer = ModelTrainer(
        model,
        device=args.device,
        checkpoint_dir=training_config.checkpoint_dir,
    )
    
    # Determine show_progress_bar
    show_progress = not args.no_progress_bar
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=training_config.num_epochs,
        learning_rate=training_config.learning_rate,
        optimizer_name=training_config.optimizer,
        loss_name=training_config.loss_fn,
        weight_decay=training_config.weight_decay,
        scheduler_name=args.lr_scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        show_progress_bar=show_progress,
        save_best_model=training_config.save_best_model,
    )
    
    # ===== TESTING =====
    print("\n[5] Evaluating on test set...")
    criterion = trainer.configure_loss(training_config.loss_fn)
    test_rmse, predictions, targets = trainer.test(test_loader, criterion)
    
    print(f"Test RMSE: {test_rmse:.6f}")
    
    # Calculate additional metrics
    mae = torch.mean(torch.abs(predictions - targets)).item()
    print(f"Test MAE: {mae:.6f}")
    
    # Calculate validation metrics from best checkpoint
    best_val_rmse = trainer.best_val_loss
    
    # Denormalize if needed
    if training_config.normalize_outputs:
        predictions_denorm = dataset.denormalize_targets(predictions.numpy())
        targets_denorm = dataset.denormalize_targets(targets.numpy())
        
        # Calculate denormalized RMSE and MAE
        denorm_rmse = torch.sqrt(torch.mean((torch.from_numpy(predictions_denorm) - torch.from_numpy(targets_denorm)) ** 2)).item()
        denorm_mae = torch.mean(torch.abs(torch.from_numpy(predictions_denorm) - torch.from_numpy(targets_denorm))).item()
        best_val_rmse_denorm = best_val_rmse * dataset.target_std
        
        print(f"Test RMSE (denormalized): {denorm_rmse:.6f}")
        print(f"Test MAE (denormalized): {denorm_mae:.6f}")
        
    else:
        best_val_rmse_denorm = best_val_rmse
        denorm_rmse_val = float(denorm_rmse)
        denorm_mae_val = float(denorm_mae)
    
    # ===== METRICS SUMMARY =====
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - TRAIN/VAL/TEST METRICS")
    print("=" * 80)
    
    # Extract best train RMSE
    best_train_rmse = min(history['train_loss']) if history['train_loss'] else float('inf')
    best_train_epoch = np.argmin(history['train_loss']) + 1 if history['train_loss'] else 0
    best_val_epoch = np.argmin(history['val_loss']) + 1 if history['val_loss'] else 0
    
    print(f"\nData Split Information:")
    print(f"  • Training samples: {len(train_loader.dataset)}")
    print(f"  • Validation samples: {len(val_loader.dataset)}")
    print(f"  • Test samples: {len(test_loader.dataset)}")
    print(f"  • Total samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
    
    print(f"\nNormalized RMSE (on 0-1 scale):")
    print(f"  • Best Training RMSE: {best_train_rmse:.6f} (at epoch {best_train_epoch})")
    print(f"  • Best Validation RMSE: {best_val_rmse:.6f} (at epoch {best_val_epoch})")
    print(f"  • Final Test RMSE: {test_rmse:.6f}")
    
    if training_config.normalize_outputs:
        print(f"\nDenormalized RMSE (in kg/m³):")
        print(f"  • Best Validation RMSE: {best_val_rmse_denorm:.2f} kg/m³")
        print(f"  • Final Test RMSE: {denorm_rmse:.2f} kg/m³")
        print(f"  • Difference (Val-Test): {best_val_rmse_denorm - denorm_rmse:.2f} kg/m³")
    
    print(f"\nMetrics Interpretation:")
    print(f"  [TRAIN RMSE] Used during training to compute gradients")
    print(f"  [VAL RMSE]   Used to select best model checkpoint (prevents overfitting)")
    print(f"  [TEST RMSE]  Used for final evaluation on unseen data (measures generalization)")
    print(f"\nOptimal scenario: Train RMSE ~= Val RMSE ~= Test RMSE")
    print(f"Large gap (Val >> Test) suggests validation set issues")
    print(f"Large gap (Val << Test) suggests overfitting to validation set")
    print("=" * 80 + "\n")
    
    # ===== SAVE RESULTS =====
    print("[6] Saving results...")
    
    # Save model config
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config.__dict__, f, indent=2)
    
    # Save training config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config.__dict__, f, indent=2)
    
    # Save normalization stats
    with open(output_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save test results
    results = {
        'test_rmse': float(test_rmse),
        'test_mae': float(mae),
        'predictions_shape': list(predictions.shape),
        'targets_shape': list(targets.shape),
        'num_test_samples': len(predictions),
    }
    
    if training_config.normalize_outputs:
        results['test_rmse_denormalized'] = float(denorm_rmse)
        results['test_mae_denormalized'] = float(denorm_mae)
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    torch.save({
        'predictions': predictions,
        'targets': targets,
    }, output_dir / "predictions.pt")
    
    print(f"* Results saved to {output_dir}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
