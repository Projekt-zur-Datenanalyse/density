"""README for Optuna Hyperparameter Optimizer

This directory contains a complete hyperparameter tuning system using Optuna
for the Chemical Density Surrogate Model.

## Overview

The Optuna hyperparameter optimizer allows you to systematically search for
optimal hyperparameter configurations for any of the 4 model architectures:

- **MLP**: Multi-Layer Perceptron
- **CNN**: Convolutional Neural Network
- **CNN_MULTISCALE**: Multi-Scale CNN (Inception-style)

## Features

- **Optuna Integration**: Uses state-of-the-art Bayesian optimization with TPE sampler
- **Cross-Platform**: Works on Windows, macOS, and Linux (no Ray dependency issues)
- **Python 3.13 Compatible**: Fully compatible with Python 3.13
- **4 Model Architectures**: Support for all 4 model types
- **Flexible Sampling**: Multiple samplers (TPE, Grid, Random)
- **Trial Pruning**: Early stopping with median pruner
- **SQLite Storage**: Persistent storage of all trials and results
- **Easy Results Export**: Export best configs for direct use with train.py

## Components

### 1. tune_config.py

Defines hyperparameter search spaces for all 4 architectures using Optuna's Trial API.

**Key Features:**

- Separate suggest function for each architecture
- Uses Optuna's `trial.suggest_*()` API for flexible parameter suggestions
- Log-scale parameters for learning rates and regularization
- Predefined search configurations (minimal, balanced, extensive, production)
- Default configurations based on config.py

**Usage:**

```python
from tune_config import OptunaSearchSpace, get_search_config

# Get suggest function for MLP
suggest_fn = OptunaSearchSpace.get_suggest_function("mlp")

# Get predefined search config
config = get_search_config("balanced")  # minimal/balanced/extensive/production
```

### 2. optuna_trainable.py

Contains objective function and model creation logic for Optuna trials.

**Key Features:**

- Creates models from hyperparameter configurations
- Trains models and returns validation RMSE
- Handles all 4 architectures
- Stores trial metadata (test RMSE, MAE, etc.)
- Error handling for failed trials

**Usage:**

- Called automatically by tune.py
- Can be customized to modify training behavior

### 3. tune.py

Main script to run hyperparameter tuning using Optuna.

**Key Features:**

- Multiple samplers: TPE (default), Grid Search, Random
- Multiple pruners: Median (default), Noop, Percentile
- Predefined search configurations
- SQLite database for persistent storage
- Automatic results saving and summary generation

**Usage:**

```bash
# Basic: Tune MLP with 50 trials
python tune.py --architecture mlp --n-trials 50

# With predefined config
python tune.py --architecture cnn --config-type balanced

# Advanced: Use different sampler/pruner
python tune.py --architecture gnn --n-trials 100 --sampler tpe --pruner median

# Parallel trials (requires thread pool)
python tune.py --architecture mlp --n-trials 100 --n-jobs 4

# All options
python tune.py --help
```

**Command Line Arguments:**

- `--architecture`: mlp, cnn, cnn_multiscale, gnn (default: mlp)
- `--n-trials`: Number of trials (default: 50)
- `--max-epochs`: Epochs per trial (default: 100)
- `--config-type`: Use predefined config (minimal/balanced/extensive/production)
- `--sampler`: tpe, grid, or random (default: tpe)
- `--pruner`: median, noop, or percentile (default: median)
- `--n-jobs`: Parallel jobs (default: 1)
- `--device`: cuda or cpu (default: auto-detect)
- `--verbose`: Enable verbose output (default: True)

### 4. optuna_analyze_results.py

Analyze and visualize tuning results from Optuna.

**Key Features:**

- Load study from SQLite database
- Display best trial metrics and hyperparameters
- Export results in multiple formats
- Generate text reports
- Generate visualization plots
- Provide command-line arguments for train.py

**Usage:**

```bash
# Show summary of latest results
python optuna_analyze_results.py

# Specific results directory
python optuna_analyze_results.py --results-dir ./optuna_results_mlp_20240115_120000

# Export and generate all outputs
python optuna_analyze_results.py --all --output-dir ./tuning_outputs

# Specific exports
python optuna_analyze_results.py --export-hyperparams --export-config
```

**Output Files:**

- `optuna_summary.json`: Best trial metrics and hyperparameters
- `best_config.json`: Best configuration for reference
- `best_hyperparams.json`: Best hyperparameters ready for train.py
- `optuna_report.txt`: Comprehensive text report
- `optuna_results.png`: Visualization of results
- `study.db`: SQLite database with all trials

## Search Space Configuration

### Predefined Configurations

The system includes 4 predefined search configurations:

1. **minimal**

   - 10 trials, 20 epochs
   - Quick testing and validation
   - Estimated time: 15-30 minutes (GPU)

2. **balanced** (default)

   - 50 trials, 50 epochs
   - Good exploration vs exploitation
   - Estimated time: 2-3 hours (GPU)

3. **extensive**

   - 100 trials, 100 epochs
   - Thorough hyperparameter exploration
   - Estimated time: 4-6 hours (GPU)

4. **production**
   - 200 trials, 150 epochs
   - Comprehensive search with full training
   - Estimated time: 12-18 hours (GPU)

### Search Space Details

#### MLP

- num_layers: 1-5
- expansion_factor: 1.0-8.0 (log scale)
- use_swiglu: True/False
- learning_rate: 1e-4 to 1e-1 (log scale)
- batch_size: 32, 64, 128, 256
- dropout_rate: 0.0-0.5
- weight_decay: 1e-6 to 1e-2 (log scale)
- optimizer: adam, sgd
- lr_scheduler: none, onecycle, cosine
- loss_fn: mse, mae

#### CNN

- cnn_expansion_size: 4, 8, 12, 16
- cnn_num_layers: 2-6
- cnn_kernel_size: 3, 5, 7
- cnn_use_batch_norm: True/False
- cnn_use_residual: True/False
- [+ common hyperparameters above]

#### CNN_MULTISCALE

- cnn_multiscale_expansion_size: 8, 12, 16, 20
- cnn_multiscale_num_scales: 2-5
- cnn_multiscale_base_channels: 8, 16, 32, 64
- [+ common hyperparameters above]

#### GNN

- gnn_hidden_dim: 8, 16, 32, 64, 128
- gnn_num_layers: 2-7
- gnn_type: gcn, gat, graphconv
- [+ common hyperparameters above]

## Workflow

### Step 1: Run Tuning

```bash
# Quick test
python tune.py --architecture mlp --config-type minimal

# Production tuning
python tune.py --architecture mlp --n-trials 100 --max-epochs 100

# With parallelization (if available)
python tune.py --architecture cnn --n-trials 100 --n-jobs 2
```

Output: `optuna_results_<architecture>_<timestamp>/` directory

### Step 2: Analyze Results

```bash
# Automatic analysis of latest results
python optuna_analyze_results.py

# Or specific directory
python optuna_analyze_results.py --results-dir ./optuna_results_mlp_20240115_120000

# Generate all outputs
python optuna_analyze_results.py --all --output-dir ./tuning_outputs
```

### Step 3: Use Best Configuration

```bash
# Export best hyperparameters
python optuna_analyze_results.py --export-hyperparams

# Use with train.py (copy the printed command)
python train.py --architecture mlp --learning-rate 0.005 --batch-size 128 ...
```

## Advantages over Ray Tune

- **Windows Compatible**: Works on Windows with Python 3.13
- **No Multi-Processing Issues**: Uses SQLite for storage instead of complex IPC
- **Simpler Setup**: No daemon processes or complex resource management
- **Better Pruning**: Median pruner for better early stopping
- **Persistent Storage**: All trials saved to SQLite database
- **Easier Debugging**: Direct access to trial database

## Output Structure

```
optuna_results_<architecture>_<timestamp>/
├── study.db                           # SQLite database with all trials
├── optuna_summary.json                # Best trial summary
└── (after analysis)
    ├── best_config.json               # Best configuration
    ├── best_hyperparams.json          # For train.py
    ├── optuna_report.txt              # Text report
    └── optuna_results.png             # Visualization
```

## Performance Optimization

### Tips for Faster Tuning

1. **Reduce Data**

   - Use a smaller subset initially for validation
   - Increase data size for final production runs

2. **Adjust Search Configuration**

   - Start with "minimal" config for quick validation
   - Scale up for production runs

3. **GPU Utilization**

   - Ensure CUDA is available
   - Single GPU is sufficient; parallelization on single machine uses threads

4. **Sampler Selection**

   - TPE (default): Good balance, Bayesian optimization
   - Random: Faster but less intelligent
   - Grid: Exhaustive but slow

5. **Pruner Selection**
   - Median: Good default, prunes slow trials early
   - Noop: No pruning, explore fully
   - Percentile: Aggressive pruning

### Troubleshooting

**GPU Out of Memory:**

- Reduce batch_size in search space
- Reduce max_epochs
- Use smaller model configurations

**Slow Progress:**

- Check device (use --device cuda)
- Reduce max_epochs for initial tests
- Use random sampler instead of TPE for faster exploration

**Failed Trials:**

- Check console output for errors
- Verify data files exist and are readable
- Check disk space for SQLite database

## Examples

### Example 1: Quick MLP Tuning

```bash
python tune.py --architecture mlp --config-type minimal
python optuna_analyze_results.py --all
```

Time: ~20 minutes

### Example 2: Comprehensive CNN Tuning

```bash
python tune.py --architecture cnn --config-type extensive --sampler tpe
python optuna_analyze_results.py --export-hyperparams
```

Time: ~4-6 hours

### Example 3: GNN with Custom Settings

```bash
python tune.py --architecture gnn \
    --n-trials 200 \
    --max-epochs 100 \
    --sampler tpe \
    --device cuda
```

Time: ~12+ hours

## Integration with train.py

After finding best hyperparameters:

```bash
# Export best hyperparameters
python optuna_analyze_results.py --export-hyperparams

# Train final model (copy command from export output)
python train.py --architecture mlp \
    --learning-rate 0.008 \
    --batch-size 128 \
    --num-layers 3 \
    --expansion-factor 5.2 \
    --epochs 200 \
    --output-dir ./final_model
```

## Database Access

Access raw trial data via SQLite:

```bash
# Query best trial
sqlite3 optuna_results_mlp_*/study.db \
  "SELECT trial_id, value FROM trials WHERE value IS NOT NULL ORDER BY value LIMIT 1;"

# Count trials
sqlite3 optuna_results_mlp_*/study.db "SELECT COUNT(*) FROM trials;"

# Export all trials to CSV
sqlite3 optuna_results_mlp_*/study.db ".mode csv" ".output trials.csv" "SELECT * FROM trials;"
```

## Notes

- Optuna uses SQLite for storage; database file grows with trials
- Each trial trains a complete model; ensure sufficient disk space
- Tuning on CPU is significantly slower; GPU strongly recommended
- Results are reproducible with same random seed (42 by default)
- SQLite database is thread-safe for parallel trials

## Troubleshooting Common Issues

### Can't find results directory

```bash
# Check what results exist
ls optuna_results_*/

# Specify directory explicitly
python optuna_analyze_results.py --results-dir ./optuna_results_mlp_20240115_120000
```

### Out of memory errors

- Reduce batch_size or max_epochs
- Use fewer parallel jobs (--n-jobs)
- Close other applications

### Plot generation fails

```bash
# Install matplotlib if missing
pip install matplotlib
python optuna_analyze_results.py --plot
```

### Database locked error

- Ensure only one script is accessing database
- Close any other Python processes
- Try deleting study.db and re-running (will lose trial history)

## Support

For issues or questions:

1. Check Optuna documentation: https://optuna.readthedocs.io/
2. Review tune_config.py for search space customization
3. Check console output for detailed error messages
4. Review SQLite database for trial metadata

"""

# Print help when run as script

if **name** == "**main**":
print(**doc**)
