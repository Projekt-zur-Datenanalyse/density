# Deep Ensemble Training - Quick Reference

## Quick Start

### Train Same-Architecture Ensemble

```python
from deep_ensemble import DeepEnsemble

# Train 5 MLP models as ensemble
ensemble = DeepEnsemble(
    architecture="mlp",
    num_models=5,
    device="cuda"
)

results = ensemble.train_ensemble(
    batch_size=32,
    max_epochs=50,
    train_ratio=0.75,
    val_ratio=0.20,
    test_ratio=0.05,
    learning_rate=0.001,
    weight_decay=1e-5
)
```

### Train Tuned Ensemble (Top-N Optuna Configs)

```python
from deep_ensemble import TunedDeepEnsemble

# Train 3 models using top 3 Optuna-tuned configs
ensemble = TunedDeepEnsemble(
    optuna_results_dir='optuna_results_mlp_20251217_120000'
)

results = ensemble.train_tuned_ensemble(
    n_models=3,
    batch_size=32,
    max_epochs=50,
    train_ratio=0.75,
    val_ratio=0.20
)

# Results saved to: tuned_ensemble_mlp_*/
```

## Key Features

| Feature                | Description                                       |
| ---------------------- | ------------------------------------------------- |
| **Fixed Test Split**   | Same test set across all models (fair comparison) |
| **Variable Train/Val** | Different splits per model (diversity)            |
| **Unique Seeds**       | Each model uses different seed for initialization |
| **Mean Prediction**    | Final prediction = mean of all models             |
| **Uncertainty**        | Std of predictions = model uncertainty            |
| **Multi-Architecture** | Mix different architectures in ensemble           |

## Classes

### DeepEnsemble

Same-architecture ensemble training.

**Constructor:**

```python
DeepEnsemble(
    architecture="mlp",        # mlp, cnn, cnn_multiscale, lightgbm
    num_models=5,              # Number of models
    device="cuda",             # cuda or cpu
    checkpoint_dir=None,       # Directory for checkpoints
    master_seed=46             # Master seed for fixed test split
)
```

**Methods:**

```python
# Train ensemble
results = ensemble.train_ensemble(
    batch_size=32,
    max_epochs=50,
    train_ratio=0.75,
    val_ratio=0.20,
    test_ratio=0.05,
    learning_rate=0.001,
    optimizer="adam",          # adam or sgd
    loss_fn="mse",            # mse or mae
    weight_decay=1e-5,
    scheduler="cosine"        # cosine or onecycle
)

# Make predictions
predictions = ensemble.predict(X_test)

# Get uncertainty
ensemble_mean, ensemble_std = ensemble.get_ensemble_predictions(X_test)

# Evaluate
metrics = ensemble.evaluate(X_test, y_test)
```

### TunedDeepEnsemble

Ensemble training using Optuna-tuned hyperparameters (rank-ordered).

**Constructor:**

```python
TunedDeepEnsemble(
    optuna_results_dir,  # Path to optuna_results_<arch>_* directory
    device="cuda"        # cuda or cpu
)
```

**Methods:**

```python
# Train ensemble with n-best tuned configs
results = ensemble.train_tuned_ensemble(
    n_models=3,           # Number of models (loads top 3 configs)
    batch_size=32,
    max_epochs=50,
    train_ratio=0.75,
    val_ratio=0.20
)

# Results include:
# - ensemble_test_rmse_denorm: Ensemble RMSE in original units
# - individual_rmses: List of each model's RMSE
# - ensemble_test_mae_denorm: Ensemble MAE
# - mean_uncertainty: Average prediction std
```

## Output Structure

### DeepEnsemble Results

```
ensemble_results/
├── ensemble_<arch>_<timestamp>/
│   ├── models/
│   │   ├── model_0.pt
│   │   ├── model_1.pt
│   │   └── ...
│   ├── ensemble_results.json      # Metrics
│   ├── ensemble_predictions.json  # Predictions + uncertainty
│   ├── training_history.json      # Per-model training history
│   └── normalization_stats.json   # Data normalization
```

### TunedDeepEnsemble Results

```
tuned_ensemble_mlp_test/
├── checkpoints_model_1/           # Model 1 checkpoints
├── checkpoints_model_2/           # Model 2 checkpoints
├── checkpoints_model_3/           # Model 3 checkpoints
├── tuned_ensemble_results.json    # Ensemble metrics
├── ensemble_configs_used.json     # Configs + seeds used
└── training_history_<model>.json  # Per-model history
```

## Result Metrics

### Individual Model Metrics

- `train_rmse`: Training RMSE (normalized)
- `val_rmse`: Validation RMSE (normalized)
- `test_rmse`: Test RMSE (normalized)
- `test_rmse_denorm`: Test RMSE in original units (kg/m³)
- `test_mae_denorm`: Test MAE in original units

### Ensemble Metrics

- `ensemble_test_rmse_denorm`: Ensemble mean prediction RMSE
- `mean_test_rmse_denorm`: Mean of individual model RMSEs
- `ensemble_test_mae_denorm`: Ensemble mean prediction MAE
- `mean_uncertainty`: Average model uncertainty (std)
- `improvement`: Ensemble vs mean improvement

## Common Workflows

### Baseline Ensemble (5 min setup)

```python
from deep_ensemble import DeepEnsemble

ensemble = DeepEnsemble(architecture="mlp", num_models=3)
results = ensemble.train_ensemble(
    batch_size=32, max_epochs=20, learning_rate=0.001
)
```

### Tuned Ensemble (After Optuna tuning)

```python
from deep_ensemble import TunedDeepEnsemble

# Must have run: python tune.py mlp --n-trials 50 --n-best-save 5
ensemble = TunedDeepEnsemble('optuna_results_mlp_20251217_120000')
results = ensemble.train_tuned_ensemble(n_models=5, max_epochs=100)
```

### Multi-Architecture Ensemble

```python
from deep_ensemble import DeepEnsemble

# Create 3 different architectures in one ensemble
# (Requires manual training of different architectures)
architectures = ["mlp", "cnn", "cnn_multiscale"]
# Train each separately, then average predictions
```

### Uncertainty Quantification

```python
ensemble = DeepEnsemble(architecture="mlp", num_models=10)
ensemble.train_ensemble()

# Get predictions with uncertainty
mean, std = ensemble.get_ensemble_predictions(X_test)

# High confidence predictions: std < 1.0 kg/m³
# Low confidence predictions: std > 5.0 kg/m³
```

### Ensemble Performance Comparison

```python
import json

# Load results
with open('ensemble_results/ensemble_mlp_*/ensemble_results.json') as f:
    results = json.load(f)

print(f"Ensemble RMSE: {results['ensemble_test_rmse_denorm']:.2f} kg/m³")
print(f"Individual avg: {results['mean_test_rmse_denorm']:.2f} kg/m³")
print(f"Improvement: {results['mean_test_rmse_denorm'] - results['ensemble_test_rmse_denorm']:.2f} kg/m³")
print(f"Uncertainty: {results['mean_uncertainty']:.4f}")
```

## Performance Tips

1. **Number of Models**: 5-10 models good balance (3 min, 10+ max)
2. **Seed Diversity**: Each model gets different seed (critical)
3. **Test Split**: Fixed across all models (ensures fair eval)
4. **Train/Val Diversity**: Variable splits create model diversity
5. **Hyperparameters**: Use tuned configs for better ensemble

## Troubleshooting

| Issue                        | Solution                                        |
| ---------------------------- | ----------------------------------------------- |
| Poor ensemble performance    | Use `TunedDeepEnsemble` instead of base configs |
| High variance between models | Check if seeds differ, increase `max_epochs`    |
| Test RMSE >> Val RMSE        | Add `weight_decay` regularization               |
| Training slow                | Use `--n-jobs` with base `DeepEnsemble`         |
| Uncertainty too high         | Increase `num_models`, increase `max_epochs`    |

## Integration with Optuna

```bash
# 1. Tune hyperparameters
python tune.py mlp --n-trials 50 --n-best-save 5

# 2. Train tuned ensemble
python -c "
from deep_ensemble import TunedDeepEnsemble
ensemble = TunedDeepEnsemble('optuna_results_mlp_20251217_120000')
ensemble.train_tuned_ensemble(n_models=5, max_epochs=100)
"

# 3. Analyze results
# Results in: tuned_ensemble_mlp_*/tuned_ensemble_results.json
```

## Next Steps

1. Run basic ensemble: `DeepEnsemble(architecture="mlp", num_models=3).train_ensemble()`
2. Run Optuna tuning: `python tune.py mlp --n-trials 50`
3. Train tuned ensemble: `TunedDeepEnsemble('optuna_results_mlp_*').train_tuned_ensemble(n_models=5)`
4. Compare results and evaluate improvement
