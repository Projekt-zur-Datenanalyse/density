# LightGBM Hyperparameter Tuning with Optuna

This guide explains how to perform hyperparameter tuning for the LightGBM model architecture using Optuna.

## Quick Start

```bash
# Tune LightGBM with default settings (50 trials, 100 epochs max)
python tune.py --architecture lightgbm

# Tune with balanced config (faster experimentation)
python tune.py --architecture lightgbm --config-type balanced

# Tune with extensive search (more comprehensive)
python tune.py --architecture lightgbm --n-trials 200 --max-epochs 150
```

## Tuning Configurations

### Predefined Configurations

Use `--config-type` to select a predefined search scope:

**Minimal** (quick testing, ~5 minutes)

```bash
python tune.py --architecture lightgbm --config-type minimal
# 10 trials, 20 max epochs per trial
```

**Balanced** (good exploration, ~30 minutes)

```bash
python tune.py --architecture lightgbm --config-type balanced
# 50 trials, 50 max epochs per trial
```

**Extensive** (thorough search, ~2 hours)

```bash
python tune.py --architecture lightgbm --config-type extensive
# 100 trials, 100 max epochs per trial
```

**Production** (comprehensive search, ~4+ hours)

```bash
python tune.py --architecture lightgbm --config-type production
# 200 trials, 150 max epochs per trial
```

## Tunable Hyperparameters

Optuna will automatically search over these LightGBM parameters:

### Tree Architecture (Controls Model Complexity)

```
lgb_num_leaves:          [7, 15, 31, 63, 127, 255]
   - Fewer leaves → simpler trees, less overfitting
   - More leaves → more complex trees, better training fit

lgb_max_depth:           [3, 5, 7, 10, 15, -1]
   - 3-5: Very shallow, strong regularization
   - 10-15: Moderate depth
   - -1: Unlimited depth (more complex)

lgb_min_child_samples:   [5, 10, 20, 30, 50]
   - Higher values → more regularization, fewer splits
   - Lower values → more splits, potential overfitting
```

### Boosting Control

```
lgb_learning_rate:       [0.001 to 0.3] (log-scale)
   - Controls step size for each boosting round
   - Lower LR needs more trees to converge
   - Higher LR trains faster but may oscillate

lgb_num_boost_round:     [50, 100, 150, 200, 300]
   - Number of trees to grow
   - Paired with learning_rate for convergence control
```

### Regularization

```
lgb_reg_alpha:           [1e-6 to 10.0] (log-scale)
   - L1 regularization on leaf weights
   - Higher → more aggressive regularization
   - Uses log-scale for fine control at low values

lgb_reg_lambda:          [1e-6 to 10.0] (log-scale)
   - L2 regularization on leaf weights
   - Higher → smoother leaf values
   - Uses log-scale for fine control at low values

lgb_subsample:           [0.5 to 1.0]
   - Fraction of samples used for each tree
   - Lower values → more stochastic, better regularization

lgb_colsample_bytree:    [0.5 to 1.0]
   - Fraction of features used for each tree
   - Lower values → feature subsampling regularization
```

### Boosting Type

```
lgb_boosting_type:       ["gbdt", "dart", "goss"]
   - gbdt: Gradient Boosting Decision Tree (standard)
   - dart: Dropouts Additive Regression Trees (more robust)
   - goss: Gradient-based One-Side Sampling (faster training)
```

## Common Tuning Strategies

### Strategy 1: Quick Exploration (5-10 minutes)

```bash
python tune.py --architecture lightgbm --config-type minimal --sampler tpe
```

Good for: Testing setup, quick baseline

### Strategy 2: Balanced Search (30-45 minutes)

```bash
python tune.py --architecture lightgbm --config-type balanced --sampler tpe --pruner median
```

Good for: Production use, reasonable search time/quality tradeoff

### Strategy 3: Regularization Focus (1 hour)

When overfitting (train RMSE << val RMSE):

```bash
python tune.py --architecture lightgbm \
  --n-trials 100 \
  --max-epochs 100 \
  --sampler tpe \
  --pruner percentile
```

Optuna will emphasize parameters: min_child_samples, reg_alpha, reg_lambda, subsample

### Strategy 4: Early Stopping Sensitivity (1.5 hours)

```bash
python tune.py --architecture lightgbm \
  --n-trials 150 \
  --max-epochs 200 \
  --sampler random \
  --pruner noop
```

Good for: Understanding early stopping effectiveness

## Sampling and Pruning Options

### Samplers

**TPE** (Tree-structured Parzen Estimator, recommended)

```bash
--sampler tpe
```

- Smart Bayesian optimization
- Balances exploration/exploitation
- Best for moderate number of trials (50-200)

**Grid** (Exhaustive search)

```bash
--sampler grid
```

- Tests all combinations
- Use with few hyperparameters
- Slow but thorough

**Random** (Random search)

```bash
--sampler random
```

- Simple baseline
- Good for very large search spaces
- Use with --n-jobs > 1 for parallelization

### Pruners

**Median** (recommended)

```bash
--pruner median
```

- Stops trials if median performance is bad
- Faster exploration
- Works well with 50+ trials

**Percentile**

```bash
--pruner percentile
```

- More aggressive pruning
- Faster but may stop promising trials early

**Noop** (No pruning)

```bash
--pruner noop
```

- Runs all trials to completion
- Slower but most accurate

## Parallel Tuning

Speed up tuning with multiple workers:

```bash
# Use 4 parallel jobs
python tune.py --architecture lightgbm \
  --n-trials 100 \
  --n-jobs 4 \
  --sampler tpe

# Use 8 parallel jobs (GPU instances recommended)
python tune.py --architecture lightgbm \
  --n-trials 200 \
  --n-jobs 8 \
  --config-type extensive
```

Note: LightGBM trains on CPU, so parallelization is highly recommended.

## Analyzing Results

After tuning, analyze the best hyperparameters:

```bash
# Show summary of best trial
python tune_analyze_results.py --show-summary

# Export best configuration
python tune_analyze_results.py --export-config

# Export hyperparameters for train.py
python tune_analyze_results.py --export-hyperparams

# Generate comprehensive report
python tune_analyze_results.py --generate-report

# All of the above
python tune_analyze_results.py --all
```

## Training with Tuned Hyperparameters

After tuning, train the final model with best hyperparameters:

```bash
# The export-hyperparams command will show how to run this:
python train.py --architecture lightgbm \
  --lgb-num-leaves <tuned_value> \
  --lgb-learning-rate <tuned_value> \
  --lgb-num-boost-round <tuned_value> \
  ... [other tuned hyperparameters]
```

Or use the exported JSON directly if you implement JSON config loading in train.py.

## Understanding Tuning Progress

The tuning output shows:

```
[Trial X] Best Value: 0.123456
  lgb_num_leaves: 31
  lgb_learning_rate: 0.05
  lgb_reg_alpha: 0.1
  ...
```

**Good signs:**

- Validation RMSE decreases steadily
- Best value improves over first 30% of trials
- Different hyperparameters being explored

**Warning signs:**

- Best value plateaus after 10 trials → too small search space
- Test RMSE >> Val RMSE → overfitting even with regularization
- Training very slow → learning_rate too small

## Troubleshooting

### Tuning takes too long

- Reduce n_trials: `--n-trials 50`
- Reduce max_epochs: `--max-epochs 50`
- Increase n_jobs: `--n-jobs 4`
- Use aggressive pruner: `--pruner percentile`

### Best trial shows poor generalization

- Add regularization constraints to search space
- Use different pruner: `--pruner noop`
- Increase val/test split in data

### All trials perform similarly

- Search space too narrow
- Try different sampler: `--sampler random`
- Run more trials to explore better

## Advanced: Custom Search Space

To modify the search space, edit `tune_config.py` and update the `suggest_lightgbm_hyperparameters` method:

```python
@staticmethod
def suggest_lightgbm_hyperparameters(trial: Trial) -> Dict[str, Any]:
    return {
        # Modify ranges here to explore different regions
        "lgb_num_leaves": trial.suggest_categorical("lgb_num_leaves", [15, 31, 63]),  # Fewer options
        # ... rest of parameters
    }
```

Then rerun tuning with your custom space.

## Example Workflows

### Workflow 1: Find Best Learning Rate

```bash
# Minimal tuning focused on learning rate
python tune.py --architecture lightgbm --n-trials 30 --max-epochs 50
# Analyze results - note which lgb_learning_rate works best
```

### Workflow 2: Regularization Parameter Search

```bash
# Focus on avoiding overfitting
python tune.py --architecture lightgbm \
  --n-trials 100 \
  --max-epochs 100 \
  --sampler tpe
# Look at reg_alpha, reg_lambda, min_child_samples in results
```

### Workflow 3: Final Production Model

```bash
# Comprehensive search for best overall performance
python tune.py --architecture lightgbm --config-type production --n-jobs 4

# Export and train final model
python tune_analyze_results.py --export-hyperparams
python train.py --architecture lightgbm [export hyperparameters]
```

## Performance Tips

1. **CPU Optimization**: LightGBM is CPU-friendly, use `--n-jobs` = number of cores
2. **Memory Usage**: Each trial loads full dataset, watch memory with many workers
3. **Early Stopping**: Tune num_boost_round and learning_rate together
4. **Feature Importance**: After tuning, check lgb feature importance for insights
5. **Ensemble**: Optuna can find diverse models useful for ensembling

## See Also

- `config.py`: Default LightGBM hyperparameters
- `lightgbm_model.py`: LightGBM model implementation
- `train.py`: Final training script using tuned hyperparameters
- `tune.py`: Main tuning script entry point
