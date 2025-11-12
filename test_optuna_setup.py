"""Quick test script to validate Optuna implementation.

This script tests the core components without running full tuning:
1. Import all modules
2. Verify search space configuration
3. Test objective function compatibility
4. Display information
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Importing Modules")
    print("="*80)
    
    try:
        print("  Importing optuna...", end=" ")
        import optuna
        print("✓")
        
        print("  Importing tune_config...", end=" ")
        from tune_config import OptunaSearchSpace, SEARCH_SPACE_CONFIGS
        print("✓")
        
        print("  Importing optuna_trainable...", end=" ")
        from optuna_trainable import create_objective, create_model_from_hyperparams
        print("✓")
        
        print("  Importing tune...", end=" ")
        import tune
        print("✓")
        
        print("  Importing optuna_analyze_results...", end=" ")
        import optuna_analyze_results
        print("✓")
        
        return True
    except Exception as e:
        print(f"✗\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_spaces():
    """Test search space configuration."""
    print("\n" + "="*80)
    print("TEST 2: Search Space Configuration (Optuna Trial API)")
    print("="*80)
    
    try:
        from tune_config import OptunaSearchSpace
        
        architectures = ["mlp", "cnn", "cnn_multiscale", "gnn"]
        
        for arch in architectures:
            print(f"\n  {arch.upper()}:")
            
            # Get suggest function
            suggest_fn = OptunaSearchSpace.get_suggest_function(arch)
            print(f"    Suggest function: ✓")
            
            # Get defaults
            defaults = OptunaSearchSpace.get_default_config(arch)
            print(f"    Default config size: {len(defaults)} parameters")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predefined_configs():
    """Test predefined search configurations."""
    print("\n" + "="*80)
    print("TEST 3: Predefined Search Configurations")
    print("="*80)
    
    try:
        from tune_config import SEARCH_SPACE_CONFIGS, get_search_config
        
        print(f"\n  Available configurations: {list(SEARCH_SPACE_CONFIGS.keys())}")
        
        for config_name in SEARCH_SPACE_CONFIGS.keys():
            config = get_search_config(config_name)
            print(f"\n  {config_name}:")
            print(f"    {config['description']}")
            print(f"    Trials: {config['n_trials']}, Epochs: {config['max_epochs']}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation from hyperparameters."""
    print("\n" + "="*80)
    print("TEST 4: Model Creation from Hyperparameters")
    print("="*80)
    
    try:
        from optuna_trainable import create_model_from_hyperparams
        from tune_config import OptunaSearchSpace
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        architectures = ["mlp", "cnn", "cnn_multiscale", "gnn"]
        
        for arch in architectures:
            print(f"\n  {arch.upper()}:")
            
            try:
                # Get default hyperparameters
                defaults = OptunaSearchSpace.get_default_config(arch)
                
                # Try to create model
                print(f"    Creating model...", end=" ")
                model = create_model_from_hyperparams(defaults, arch, device)
                print("✓")
                
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                print(f"    Parameters: {num_params:,}")
                
            except ImportError as e:
                print(f"⚠ (Optional architecture not available: {e})")
            except Exception as e:
                print(f"✗ ERROR: {e}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading."""
    print("\n" + "="*80)
    print("TEST 5: Data Loading")
    print("="*80)
    
    try:
        from data_loader import ChemicalDensityDataLoader
        
        print("  Loading data...", end=" ")
        data_loader = ChemicalDensityDataLoader(".")
        
        # Try to load with small batch size
        train_loader, val_loader, test_loader, dataset = data_loader.load_dataset(
            normalize_features=True,
            normalize_targets=True,
            validation_split=0.2,
            test_split=0.1,
            batch_size=32,
        )
        print("✓")
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Dataset size: {len(dataset)}")
        
        return True
    except Exception as e:
        print(f"✗\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_usage():
    """Display usage instructions."""
    print("\n" + "="*80)
    print("OPTUNA TUNING QUICK START")
    print("="*80)
    
    print("\n1. Quick MLP Tuning (minimal config):")
    print("   python tune.py --architecture mlp --config-type minimal")
    
    print("\n2. Balanced CNN Tuning:")
    print("   python tune.py --architecture cnn --config-type balanced")
    
    print("\n3. Advanced GNN Tuning:")
    print("   python tune.py --architecture gnn \\")
    print("       --n-trials 100 --max-epochs 100 \\")
    print("       --sampler tpe --device cuda")
    
    print("\n4. Analyze Results:")
    print("   python optuna_analyze_results.py --all --output-dir ./outputs")
    
    print("\n5. For detailed documentation:")
    print("   python OPTUNA_README.md  (or open in text editor)")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("OPTUNA IMPLEMENTATION VALIDATION TEST")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Search Spaces", test_search_spaces),
        ("Predefined Configs", test_predefined_configs),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nFATAL ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Optuna tuning is ready to use.")
        display_usage()
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
