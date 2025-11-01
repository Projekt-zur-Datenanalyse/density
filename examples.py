"""
Example usage of the Chemical Density Surrogate Model.

This script shows how to:
1. Create a model with different configurations
2. Make predictions with a trained model
3. Experiment with different architectures
"""

import torch
from config import ModelConfig
from model import ChemicalDensitySurrogate


def example_basic_usage():
    """Example 1: Basic model creation and forward pass."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Model Creation and Forward Pass")
    print("=" * 70)
    
    # Create a default configuration
    config = ModelConfig(
        num_layers=1,
        expansion_factor=100,
        use_swiglu=True,
    )
    
    # Create the model
    model = ChemicalDensitySurrogate(config)
    print(model.get_model_info())
    
    # Create sample input: 4 features [SigC, SigH, EpsC, EpsH]
    batch_size = 8
    sample_input = torch.randn(batch_size, 4)
    
    print(f"\nInput shape: {sample_input.shape}")
    
    # Forward pass
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (densities): {output.squeeze().tolist()}")


def example_swish_activation():
    """Example 2: Model with Swish instead of SwiGLU."""
    print("\n" + "=" * 70)
    print("Example 2: Model with Swish Activation")
    print("=" * 70)
    
    config = ModelConfig(
        num_layers=1,
        expansion_factor=100,
        use_swiglu=False,  # Use Swish instead
    )
    
    model = ChemicalDensitySurrogate(config)
    print(model.get_model_info())
    
    # Test forward pass
    sample_input = torch.randn(16, 4)
    output = model(sample_input)
    print(f"\nPredicted densities: {output.squeeze().tolist()}")


def example_multi_layer():
    """Example 3: Multi-layer model with residual connections."""
    print("\n" + "=" * 70)
    print("Example 3: Multi-Layer Model (with residual connections)")
    print("=" * 70)
    
    config = ModelConfig(
        num_layers=3,  # 3 hidden layers with residuals
        expansion_factor=100,
        use_swiglu=True,
    )
    
    model = ChemicalDensitySurrogate(config)
    print(model.get_model_info())
    
    # Test forward pass
    sample_input = torch.randn(32, 4)
    output = model(sample_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Number of predictions: {len(output)}")


def example_different_expansions():
    """Example 4: Comparing different expansion factors."""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Different Expansion Factors")
    print("=" * 70)
    
    expansion_factors = [50, 100, 200]
    
    for factor in expansion_factors:
        config = ModelConfig(
            num_layers=1,
            expansion_factor=factor,
            use_swiglu=True,
        )
        
        model = ChemicalDensitySurrogate(config)
        params = model.get_num_parameters()
        
        print(f"\nExpansion Factor: {factor}x")
        print(f"  Hidden Dim: {config.get_hidden_dim()}")
        print(f"  Parameters: {params:,}")


def example_cuda_support():
    """Example 5: CUDA support and device placement."""
    print("\n" + "=" * 70)
    print("Example 5: CUDA Support")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")
    
    config = ModelConfig(
        num_layers=2,
        expansion_factor=100,
        use_swiglu=True,
        device=device,
    )
    
    model = ChemicalDensitySurrogate(config)
    model.to(device)
    
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create sample input on device
    sample_input = torch.randn(4, 4, device=device)
    output = model(sample_input)
    
    print(f"Output device: {output.device}")
    print(f"Output values: {output.squeeze().tolist()}")


def example_batch_prediction():
    """Example 6: Making predictions on large batches."""
    print("\n" + "=" * 70)
    print("Example 6: Batch Prediction")
    print("=" * 70)
    
    config = ModelConfig(
        num_layers=2,
        expansion_factor=100,
        use_swiglu=True,
    )
    
    model = ChemicalDensitySurrogate(config)
    model.eval()
    
    # Create a large batch of data
    num_samples = 1000
    features = torch.randn(num_samples, 4)
    
    print(f"Making predictions on {num_samples} samples...")
    
    # Predict in batches
    batch_size = 100
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = features[i:i+batch_size]
            predictions = model(batch)
            all_predictions.append(predictions)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Mean prediction: {all_predictions.mean().item():.6f}")
    print(f"Std prediction: {all_predictions.std().item():.6f}")
    print(f"Min prediction: {all_predictions.min().item():.6f}")
    print(f"Max prediction: {all_predictions.max().item():.6f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CHEMICAL DENSITY SURROGATE MODEL - USAGE EXAMPLES")
    print("=" * 70)
    
    example_basic_usage()
    example_swish_activation()
    example_multi_layer()
    example_different_expansions()
    example_cuda_support()
    example_batch_prediction()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
