# Deep Ensemble Analysis Script

A comprehensive visualization and analysis tool for deep ensemble results.

## Overview

The `analyze_ensemble.py` script loads trained deep ensemble checkpoints, generates predictions, and creates detailed visualizations showing model performance, uncertainty quantification, and ensemble benefits.

## Usage

```bash
# Analyze tuned ensemble results
python analyze_ensemble.py --results-dir tuned_ensemble_results

# With custom output directory
python analyze_ensemble.py --results-dir tuned_ensemble_results --output-dir my_analysis

# Use CPU for inference (if CUDA not available)
python analyze_ensemble.py --results-dir tuned_ensemble_results --device cpu
```

## Output Visualizations

The script generates 5 comprehensive PNG plots and a detailed text report:

### 1. **01_accuracy_comparison.png**

- **Left panel**: Individual model RMSE vs ensemble RMSE
  - Shows performance of each model with ensemble comparison
  - Red dashed line indicates mean individual performance
  - Green bar highlights ensemble advantage
- **Right panel**: Improvement metrics
  - Mean RMSE, ensemble RMSE, and net improvement
  - Quantifies the ensemble benefit

**Use case**: Quick overview of whether the ensemble outperforms individual models.

---

### 2. **02_predictions_vs_true.png**

- **Uncertainty bands**: Min-max range across all models (light blue)
- **Ensemble uncertainty**: ±1 standard deviation around ensemble predictions (orange)
- **Ensemble prediction**: Red line showing mean prediction
- **True values**: Green line showing ground truth
- **Individual models**: Faint blue lines showing each model's prediction

**Use case**: Visualize where predictions are confident vs uncertain, and whether true values fall within uncertainty bands.

---

### 3. **03_error_distributions.png**

4-panel visualization:

- **Top-left**: Ensemble error distribution histogram
  - Shows bias and spread of ensemble errors
  - Red dashed line at zero, orange dashed line at mean
- **Top-right**: Individual model error distributions (overlaid)
  - Compare error patterns across models
  - Lighter colors = different models
- **Bottom-left**: Box plot of absolute errors
  - Color-coded: Light blue for individual models, dark green for ensemble
  - Shows median, quartiles, and outliers
- **Bottom-right**: Cumulative error distribution
  - X-axis: Absolute error magnitude
  - Y-axis: Percentage of samples with error ≤ X
  - Ensemble vs first 3 models for clarity

**Use case**: Understand error characteristics - is the ensemble more robust? Are there systematic biases?

---

### 4. **04_uncertainty_analysis.png**

4-panel uncertainty visualization:

- **Top-left**: Uncertainty vs Absolute Error (scatter)
  - Shows correlation between uncertainty and actual error
  - Points colored by true value
  - Red dashed trend line
  - **Goal**: Higher correlation = uncertainty is predictive of error
- **Top-right**: Uncertainty vs True Value (scatter)
  - Shows if certain regions of the output space are harder to predict
  - Points colored by absolute error
- **Bottom-left**: Uncertainty histogram
  - Shows distribution of model disagreement
  - Red dashed = mean, orange dashed = median
- **Bottom-right**: Residuals (signed error) vs Uncertainty
  - Shows whether uncertainty captures both overestimation and underestimation
  - Should be symmetric around zero if well-calibrated

**Use case**: Evaluate quality of uncertainty quantification. Are predictions more uncertain when they're actually wrong?

---

### 5. **05_model_comparison.png**

4-panel per-model metrics:

- **Top-left**: RMSE per model (with mean line)
- **Top-right**: MAE per model (with mean line)
- **Bottom-left**: R² score per model (with mean line)
- **Bottom-right**: Prediction bias per model
  - Red bars: underestimation (negative bias)
  - Green bars: overestimation (positive bias)
  - Black line at zero
  - Orange dashed: mean bias

**Use case**: Identify which models perform best and whether any have systematic biases.

---

### 6. **analysis_report.txt**

Detailed text report containing:

**Sections:**

- **Ensemble Performance**: RMSE, MAE, improvement metrics
- **Uncertainty Metrics**: Mean and std of model disagreement
- **Individual Model Performance**: Per-model RMSE and MAE
- **Error Statistics**: Min, max, mean, median, 95th percentile errors
- **Uncertainty Statistics**: Distribution of uncertainty values
- **Correlation Analysis**: Uncertainty vs error correlation
- **Summary**: Interpretation and quality assessment

**Key metrics in report:**

- `Improvement (%)`: How much better is ensemble vs average model
- `Uncertainty vs Error Correlation`: How predictive is uncertainty of actual error
  - 0.4-0.6 = moderate calibration
  - 0.6+ = good calibration
  - < 0.3 = poor calibration

## Interpreting Results

### Good Signs ✓

- Ensemble RMSE lower than mean individual RMSE
- Uncertainty increases when predictions are wrong (high correlation in 04_uncertainty_analysis)
- Error distributions are roughly symmetric (centered near 0)
- Individual models have different prediction patterns (different colors in 02_predictions_vs_true)
- R² scores > 0.8 (in 05_model_comparison)

### Things to Watch ⚠️

- Very low improvement % (< 1%) = models too similar, ensemble not diverse
- Negative correlation in 04 top-left = uncertainty not predictive of error
- Extreme outliers in 03_error_distributions = systematic failures on some data
- Scattered bias values in 05 = individual models have inconsistent errors

## Arguments

```
--results-dir DIR          Path to ensemble results directory (required)
                          Must contain tuned_ensemble_results.json and checkpoints

--output-dir DIR          Output directory for visualizations
                          Default: ensemble_analysis

--device {cuda, cpu}      Device for inference
                          Default: cuda (if available)
```

## Requirements

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn

Automatically imports from your project:

- `deep_ensemble.load_data_with_fixed_test_split()`
- `optuna_trainable.create_model_from_hyperparams()`

## File Structure Expected

```
results_directory/
├── tuned_ensemble_results.json       # Ensemble metrics
├── ensemble_configs_used.json        # Model configs
└── checkpoints_model_1/
    ├── checkpoints_model_2/
    ├── checkpoints_model_3/
    └── ...
        └── best_model.pt             # Saved model weights
```

## Examples

### After tuned ensemble training:

```bash
# Train ensemble
python deep_ensemble.py --optuna-results optuna_results_mlp_* --n-models 8

# Analyze
python analyze_ensemble.py --results-dir tuned_ensemble_results --output-dir my_analysis

# Results:
# - 5 PNG plots showing all aspects of ensemble behavior
# - Text report with detailed metrics and interpretation
```

## Tips

1. **Reproducibility**: Uses same test split and seeds as training
2. **Memory**: Loads all models into memory - for large ensembles on small GPU, use `--device cpu`
3. **Interpretation**: Start with 01_accuracy_comparison and 04_uncertainty_analysis
4. **Comparison**: Run with different ensemble configs and compare outputs to find optimal n_models

## See Also

- [DEEP_ENSEMBLE_QUICK_REFERENCE.md](DEEP_ENSEMBLE_QUICK_REFERENCE.md) - How to train ensembles
- [OPTUNA_QUICK_REFERENCE.md](OPTUNA_QUICK_REFERENCE.md) - How to tune hyperparameters
- [deep_ensemble.py](deep_ensemble.py) - Training script
- [tune.py](tune.py) - Hyperparameter tuning script
