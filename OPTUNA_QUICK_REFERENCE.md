# Optuna Hyperparameter Tuning - Quick Reference

## Quick Start

```bash
# Basic: Tune MLP with default settings
python tune.py mlp

# Tune with n-best configs saved
python tune.py mlp --n-trials 50 --n-best-save 5

# Tune CNN with specific epochs
python tune.py cnn --n-trials 100 --max-epochs 100

# Use predefined search configuration
python tune.py lightgbm --config-type balanced
```

## Command Line Arguments

| Argument        | Values                                | Default  | Description                     |
| --------------- | ------------------------------------- | -------- | ------------------------------- |
| `architecture`  | mlp, cnn, cnn_multiscale, lightgbm    | -        | Model to tune (positional)      |
| `--n-trials`    | integer                               | 50       | Number of optimization trials   |
| `--max-epochs`  | integer                               | 100      | Epochs per trial                |
| `--n-best-save` | integer                               | 5        | Number of best configs to save  |
| `--config-type` | minimal/balanced/extensive/production | balanced | Predefined search config        |
| `--sampler`     | tpe/grid/random                       | tpe      | Optimization sampler            |
| `--pruner`      | median/noop/percentile                | median   | Early stopping strategy         |
| `--n-jobs`      | integer                               | 1        | Parallel jobs                   |
| `--device`      | cuda/cpu                              | auto     | Compute device                  |
| `--seed`        | integer                               | 46       | Master seed for reproducibility |

## Predefined Configurations

```bash
# Quick test (10 trials, 20 epochs)
python tune.py mlp --config-type minimal

# Standard tuning (50 trials, 50 epochs) - RECOMMENDED
python tune.py mlp --config-type balanced

# Comprehensive search (100 trials, 100 epochs)
python tune.py mlp --config-type extensive

# Production (200 trials, 150 epochs)
python tune.py mlp --config-type production
```

## Samplers

- **tpe** (default): Bayesian optimization, best for 50-200 trials
- **grid**: Exhaustive search, use with few parameters
- **random**: Simple baseline, good with `--n-jobs > 1`

## Pruners

- **median** (default): Stops poor trials early, fast exploration
- **percentile**: More aggressive pruning, faster but risky
- **noop**: No pruning, all trials run to completion

## Output

Results saved to: `optuna_results_<arch>_<timestamp>/`

Contents:

- `configs/config_rank_*.json` - Top N configurations with full metadata
- `optuna_metadata.json` - Study metadata and summary
- `study.db` - SQLite database with all trial results

## Analyzing Results

```python
from optuna_manager import OptunaAnalyzer, OptunaVisualizer

# Load latest results
analyzer = OptunaAnalyzer('optuna_results_mlp_20251217_120000')

# Display summary
analyzer.display_summary()

# Export best config
best_config = analyzer.export_best_config('best_config.json')

# Generate report
report = analyzer.generate_report('tuning_report.txt')

# Visualize
visualizer = OptunaVisualizer('optuna_results_mlp_20251217_120000')
visualizer.plot_optimization_history('history.png')
visualizer.plot_parameter_importance('importance.png')
```

## Using Tuned Configs

```python
from deep_ensemble import TunedDeepEnsemble

# Train ensemble with top 3 tuned configs
ensemble = TunedDeepEnsemble('optuna_results_mlp_20251217_120000')
ensemble.train_tuned_ensemble(n_models=3, batch_size=32, max_epochs=50)

# Results saved to: tuned_ensemble_mlp_*/
```

## Seed Management

- **Master seed**: 46 (configurable via `--seed`)
- **Trial seeds**: Generated uniquely per trial from master seed
- **Reproducibility**: Same seed + config = identical results

## Performance Tips

1. **GPU tuning**: Use `--device cuda` and `--n-jobs 1-2`
2. **CPU tuning**: Use `--n-jobs 4-8` for parallelization
3. **Quick results**: Use `--config-type minimal` first
4. **Production**: Use `--config-type extensive` or `--config-type production`
5. **Regularization focus**: Use `--n-trials 100` with `--pruner percentile`

## Common Workflows

### Find Best Baseline (15 min)

```bash
python tune.py mlp --config-type minimal
```

### Production Tuning (2-3 hours)

```bash
python tune.py cnn --n-trials 100 --max-epochs 100 --n-best-save 10
```

### Compare Architectures

```bash
python tune.py mlp --config-type balanced
python tune.py cnn --config-type balanced
python tune.py cnn_multiscale --config-type balanced
python tune.py lightgbm --config-type balanced
```

### Parallel Tuning (30 min with 4 GPUs)

```bash
python tune.py mlp --n-trials 100 --n-jobs 4 --sampler tpe
```

## Troubleshooting

| Issue              | Solution                                                            |
| ------------------ | ------------------------------------------------------------------- |
| Tuning too slow    | Use `--config-type minimal`, `--n-jobs 4`, or `--pruner percentile` |
| Poor results       | Use `--config-type extensive`, increase `--n-trials`                |
| All trials similar | Use `--sampler random`, increase search space                       |
| High variance      | Use `--pruner noop` for complete trials                             |

## Next Steps

1. Run tuning: `python tune.py <architecture> --n-trials 50`
2. Analyze results: Load `optuna_results_*/` directory
3. Train ensemble: Use top-n configs with `TunedDeepEnsemble`
4. Evaluate: Compare performance metrics
