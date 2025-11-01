"""
Unified training script supporting multiple architectures: MLP, GNN, and CNN.
Allows easy comparison of different model types on the same data.
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from datetime import datetime
import sys

from data_loader import ChemicalDensityDataLoader
from config import ModelConfig, TrainingConfig
from trainer import ModelTrainer
from model import ChemicalDensitySurrogate

# Import optional models
try:
    from gnn_model import GraphNeuralSurrogate
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("Warning: GNN model not available (torch_geometric may not be installed)")

try:
    from cnn_model import ConvolutionalSurrogate, MultiScaleConvolutionalSurrogate
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("Warning: CNN models not available")


def create_model(model_type: str, num_features: int, config: ModelConfig) -> nn.Module:
    """Create a model of the specified type.
    
    Args:
        model_type: Type of model ('mlp', 'gnn', 'cnn', 'cnn_multiscale')
        num_features: Number of input features
        config: Model configuration
    
    Returns:
        Initialized model
    """
    if model_type == 'mlp':
        # Update config with correct input dimensions
        config.input_dim = num_features
        return ChemicalDensitySurrogate(config)
    
    elif model_type == 'gnn':
        if not GNN_AVAILABLE:
            raise RuntimeError(
                "GNN model requires torch_geometric. "
                "Install with: pip install torch_geometric"
            )
        return GraphNeuralSurrogate(
            num_input_features=num_features,
            hidden_dim=64,
            num_gnn_layers=3,
            gnn_type='gcn',
            dropout_rate=config.dropout_rate,
        )
    
    elif model_type == 'cnn':
        return ConvolutionalSurrogate(
            num_input_features=num_features,
            expansion_size=8,
            num_conv_layers=4,
            kernel_size=3,
            use_batch_norm=True,
            use_residual=True,
            dropout_rate=config.dropout_rate,
        )
    
    elif model_type == 'cnn_multiscale':
        return MultiScaleConvolutionalSurrogate(
            num_input_features=num_features,
            expansion_size=8,
            num_scales=3,
            base_channels=16,
            dropout_rate=config.dropout_rate,
        )
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: mlp, gnn, cnn, cnn_multiscale"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Train multiple model architectures on chemical density prediction'
    )
    
    # Model selection
    parser.add_argument(
        '--model-type',
        type=str,
        default='mlp',
        choices=['mlp', 'gnn', 'cnn', 'cnn_multiscale'],
        help='Type of model architecture to train'
    )
    
    # Architecture parameters (for MLP)
    parser.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='Number of hidden layers (MLP only)'
    )
    parser.add_argument(
        '--expansion-factor',
        type=int,
        default=None,
        help='Expansion factor for hidden layers (MLP only)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='Weight decay for L2 regularization'
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=None,
        help='Dropout rate'
    )
    
    # Scheduler
    parser.add_argument(
        '--scheduler',
        type=str,
        default='none',
        choices=['none', 'onecycle', 'cosine'],
        help='Learning rate scheduler'
    )
    
    # Output
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output files'
    )
    parser.add_argument(
        '--save-best',
        action='store_true',
        help='Save best model checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ModelConfig()
    train_config = TrainingConfig()
    
    # Override config with CLI arguments
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.expansion_factor is not None:
        config.expansion_factor = args.expansion_factor
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    
    if args.epochs is not None:
        train_config.num_epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        train_config.weight_decay = args.weight_decay
    
    print("\n" + "="*70)
    print(f"Training {args.model_type.upper()} Model")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_loader = ChemicalDensityDataLoader(data_dir='.')
    train_loader, val_loader, test_loader, _ = data_loader.load_dataset(
        batch_size=train_config.batch_size,
    )
    
    # Get dataset info
    num_features = 4  # SigC, SigH, EpsC, EpsH
    total_samples = len(data_loader.load_dataset()[3])
    print(f"Number of input features: {num_features}")
    print(f"Total samples: {total_samples}")
    
    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    model = create_model(args.model_type, num_features, config)
    
    if hasattr(model, 'get_model_info'):
        print(model.get_model_info())
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
    )
    
    # Train
    print(f"\nTraining for {train_config.num_epochs} epochs...")
    print("-" * 70)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_config.num_epochs,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        scheduler_name=args.scheduler,
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)
    
    criterion = nn.MSELoss()
    
    # Test evaluation
    test_rmse, test_preds, test_targets = trainer.test(test_loader, criterion)
    test_mse = criterion(test_preds, test_targets).item()
    
    # Validation evaluation
    val_rmse, val_preds, val_targets = trainer.test(val_loader, criterion)
    val_mse = criterion(val_preds, val_targets).item()
    
    # Denormalize results
    # Get denormalization stats from data loader
    _, _, _, full_dataset = data_loader.load_dataset()
    
    # Use target stats from dataset
    target_mean = full_dataset.target_mean if hasattr(full_dataset, 'target_mean') else 604.0
    target_std = full_dataset.target_std if hasattr(full_dataset, 'target_std') else 201.0
    
    val_preds_denorm = val_preds * target_std + target_mean
    val_targets_denorm = val_targets * target_std + target_mean
    val_rmse_denorm = torch.sqrt(torch.mean((val_preds_denorm - val_targets_denorm) ** 2)).item()
    val_mae_denorm = torch.mean(torch.abs(val_preds_denorm - val_targets_denorm)).item()
    
    test_preds_denorm = test_preds * target_std + target_mean
    test_targets_denorm = test_targets * target_std + target_mean
    test_rmse_denorm = torch.sqrt(torch.mean((test_preds_denorm - test_targets_denorm) ** 2)).item()
    test_mae_denorm = torch.mean(torch.abs(test_preds_denorm - test_targets_denorm)).item()
    
    print(f"\nValidation Loss (MSE): {val_mse:.6f}")
    print(f"Validation RMSE (normalized): {val_rmse:.6f}")
    print(f"Validation RMSE (denormalized): {val_rmse_denorm:.2f} kg/m³")
    print(f"Validation MAE (denormalized): {val_mae_denorm:.2f} kg/m³")
    
    print(f"\nTest Loss (MSE): {test_mse:.6f}")
    print(f"Test RMSE (normalized): {test_rmse:.6f}")
    print(f"Test RMSE (denormalized): {test_rmse_denorm:.2f} kg/m³")
    print(f"Test MAE (denormalized): {test_mae_denorm:.2f} kg/m³")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output_prefix if args.output_prefix else f"{args.model_type}_{timestamp}"
    
    results = {
        'model_type': args.model_type,
        'timestamp': timestamp,
        'config': {
            'num_layers': config.num_layers if args.model_type == 'mlp' else None,
            'expansion_factor': config.expansion_factor if args.model_type == 'mlp' else None,
            'dropout_rate': config.dropout_rate,
        },
        'training_config': {
            'num_epochs': train_config.num_epochs,
            'batch_size': train_config.batch_size,
            'learning_rate': train_config.learning_rate,
            'weight_decay': train_config.weight_decay,
            'scheduler': args.scheduler,
        },
        'model_params': sum(p.numel() for p in model.parameters()),
        'results': {
            'val_loss': float(val_mse),
            'val_rmse_norm': float(val_rmse),
            'val_rmse_denorm': float(val_rmse_denorm),
            'val_mae_denorm': float(val_mae_denorm),
            'test_loss': float(test_mse),
            'test_rmse_norm': float(test_rmse),
            'test_rmse_denorm': float(test_rmse_denorm),
            'test_mae_denorm': float(test_mae_denorm),
        }
    }
    
    # Save results JSON
    results_file = f"results_{output_prefix}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save training history
    history_file = f"history_{output_prefix}.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_file}")
    
    # Save best model if requested
    if args.save_best:
        model_file = f"model_best_{output_prefix}.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Best model saved to: {model_file}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    main()
