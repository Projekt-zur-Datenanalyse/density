# Complete Metrics Explanation: MSE, RMSE, MAE, and R²

## Overview

This document explains how metrics are calculated in the Chemical Density Surrogate Model, including the often-missing **R² (coefficient of determination)**.

---

## 1. MSE/RMSE Calculation Flow

### Where MSE is Calculated

**File**: `trainer.py` (lines 145-227)

**Methods**:

- `train_epoch()` - Line 150-191
- `validate()` - Line 193-227
- `test()` - Line 365-402

### Step-by-Step RMSE Calculation

```python
# In trainer.py, train_epoch() method:

1. For each batch:
   predictions = model(features)              # Model forward pass
   loss = criterion(predictions, targets)     # Loss calculation (MSE)
   total_loss += loss.item()                  # Accumulate MSE

2. After all batches:
   avg_mse = total_loss / num_batches         # Average MSE (on normalized scale)
   rmse = sqrt(avg_mse)                       # Convert to RMSE (on normalized scale)
   return rmse
```

### Key Formula

$$\text{MSE}_{\text{normalized}} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{pred,norm}} - y_{\text{true,norm}})^2$$

$$\text{RMSE}_{\text{normalized}} = \sqrt{\text{MSE}_{\text{normalized}}}$$

---

## 2. Denormalization Process

### Where Denormalization Happens

**File**: `optuna_trainable.py` (lines 208-219)

```python
# Line 208: Get test RMSE (on normalized scale)
test_rmse, predictions, targets = trainer.test(test_loader, criterion)

# Line 213-216: Denormalize RMSE and MAE
# Formula: error_denorm = error_norm × target_std
test_rmse_denorm = test_rmse * test_loader.dataset.target_std
mae_denorm = mae * test_loader.dataset.target_std

# Return both normalized and denormalized
return best_val_loss, test_rmse, test_rmse_denorm, mae_denorm
```

### Why This Works

The targets are normalized as:
$$y_{\text{norm}} = \frac{y_{\text{true}} - \text{mean}}{\text{std}}$$

To reverse this for errors (differences, not absolute values):
$$\text{RMSE}_{\text{denorm}} = \text{RMSE}_{\text{norm}} \times \text{std}$$

**Note**: We only multiply by std, NOT add mean, because RMSE is an error metric (a difference), not an absolute value.

### Example

```
Dataset statistics:
- target_mean = 673.27 kg/m³
- target_std = 104.26 kg/m³

Model output:
- test_rmse (normalized) = 0.3973
- test_rmse (denormalized) = 0.3973 × 104.26 = 41.46 kg/m³

Interpretation: Average prediction error is ±41.46 kg/m³
```

---

## 3. Where R² Should Be (But Isn't)

### What R² Measures

R² (coefficient of determination) tells you what **fraction of variance** the model explains:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

Where:

- $\text{SS}_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2$ = Sum of squared residuals
- $\text{SS}_{\text{tot}} = \sum_i (y_i - \bar{y})^2$ = Total sum of squares

### Interpretation

| R² Value | Meaning                                     |
| -------- | ------------------------------------------- |
| 1.0      | Perfect predictions                         |
| 0.9      | Model explains 90% of variance              |
| 0.6      | Model explains 60% of variance              |
| 0.0      | Model explains 0% (predicts mean always)    |
| < 0      | Model worse than baseline (predicting mean) |

---

## 4. Current Metrics in the Codebase

### What We Currently Track

1. **Validation RMSE (normalized)** - Used for trial pruning
2. **Test RMSE (normalized)** - Stored in Optuna trial
3. **Test RMSE (denormalized)** - Stored in Optuna trial user_attrs
4. **Test MAE (denormalized)** - Stored in Optuna trial user_attrs

### What We're Missing

1. ❌ **R² Score** - No R² calculation anywhere
2. ❌ **Relative Accuracy** - We calculate this in `optuna_analyze_results.py` but not here
3. ❌ **Baseline Comparison** - No comparison to mean-predictor baseline

### Where Relative Accuracy IS Calculated

**File**: `optuna_analyze_results.py` (display_summary method)

```python
relative_accuracy = (1.0 - test_rmse_denorm / self.target_std) * 100.0
```

This is essentially: `R² × 100%` (but only when using denormalized RMSE).

---

## 5. Complete Metric Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING PHASE (trainer.py)                                 │
│ ─────────────────────────────────────────────────────────── │
│ Input: Normalized features & targets                         │
│ Model Forward: predictions (normalized scale)                │
│ Loss Function: MSE(pred_norm, target_norm)                   │
│ Output: RMSE_normalized = sqrt(avg_mse)                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ TESTING PHASE (trainer.test)                                │
│ ─────────────────────────────────────────────────────────── │
│ Collect all predictions and targets on test set              │
│ Calculate: RMSE_normalized = sqrt(avg_mse)                   │
│ Also capture raw predictions & targets for later use         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ DENORMALIZATION (optuna_trainable.py)                       │
│ ─────────────────────────────────────────────────────────── │
│ RMSE_denorm = RMSE_norm × target_std                        │
│ MAE_denorm = MAE_norm × target_std                          │
│ Store both normalized and denormalized values                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ANALYSIS (optuna_analyze_results.py)                        │
│ ─────────────────────────────────────────────────────────── │
│ Load denormalized RMSE                                       │
│ Calculate: RelativeAccuracy = (1 - RMSE_denorm/std) × 100%  │
│ This is R² × 100%                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Adding R² Calculation

### Proposed Implementation

Add to `trainer.py` test method:

```python
def test(self, test_loader: DataLoader, criterion):
    """Evaluate on test set with R² calculation."""
    self.model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0).squeeze()

    # Calculate metrics
    avg_mse = total_loss / len(test_loader)
    avg_rmse = torch.sqrt(torch.tensor(avg_mse)).item()

    # Calculate R² (coefficient of determination)
    ss_res = torch.sum((all_targets - all_predictions) ** 2)      # SS_residual
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2)   # SS_total
    r2_score = 1.0 - (ss_res / ss_tot).item()

    # Calculate MAE
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()

    return avg_rmse, all_predictions, all_targets, r2_score, mae
```

### Add to optuna_trainable.py:

```python
test_rmse, predictions, targets, r2_score, mae = trainer.test(test_loader, criterion)

# Denormalize
test_rmse_denorm = test_rmse * test_loader.dataset.target_std
mae_denorm = mae * test_loader.dataset.target_std

# Store R² in trial
trial.set_user_attr("r2_score", r2_score)
trial.set_user_attr("r2_pct", r2_score * 100)
```

---

## 7. Current Metric Values vs. Full Picture

### What We Report Now

```
Best Validation RMSE: 0.3973
Best Test RMSE (denorm): 41.46 kg/m³
Best Test MAE (denorm): 12.34 kg/m³
Target Std Dev: 104.26 kg/m³
```

### What We Should Also Report

```
R² Score: 0.8475 (84.75% of variance explained)
Relative Accuracy: 60.25% (1 - 41.46/104.26) × 100
Baseline Error: 104.26 kg/m³ (always predicting mean)
Model Improvement: 60.25% better than baseline
```

---

## 8. Understanding the Relationship Between Metrics

### All These Are Equivalent (In Percentage Terms)

```python
# Using denormalized RMSE
relative_accuracy = (1 - rmse_denorm / target_std) * 100
r2_score_pct = relative_accuracy
rmse_pct_of_std = (rmse_denorm / target_std) * 100

# So:
# relative_accuracy + rmse_pct_of_std = 100%
# r2_score + rmse_pct = 100%
```

### Example with Real Numbers

```
If:
- target_std = 104.26 kg/m³
- rmse_denorm = 41.46 kg/m³

Then:
- R² = 1 - (41.46 / 104.26)² / (1)² ≠ simple ratio (this is wrong!)
- Relative Accuracy = (1 - 41.46/104.26) × 100 = 60.25%
- RMSE % of Std = (41.46 / 104.26) × 100 = 39.75%
- Sum check: 60.25% + 39.75% = 100% ✓

Note: R² is calculated from variance (squared), not absolute RMSE!
```

---

## 9. Summary Table

| Metric            | Where Calculated          | Scale                  | Interpretation            |
| ----------------- | ------------------------- | ---------------------- | ------------------------- |
| MSE               | trainer.py:train_epoch    | Normalized             | Loss value to minimize    |
| RMSE (norm)       | trainer.py:validate, test | Normalized (0-1ish)    | Error on normalized scale |
| RMSE (denorm)     | optuna_trainable.py       | Original units (kg/m³) | Actual prediction error   |
| MAE               | optuna_trainable.py       | Original units (kg/m³) | Median error magnitude    |
| **R² (missing)**  | **Nowhere**               | **Percentage (0-1)**   | **Variance explained**    |
| Relative Accuracy | optuna_analyze_results.py | Percentage (0-100)     | Same as R² × 100          |

---

## 10. Recommendations

### Immediate Fix

1. Calculate R² in `trainer.test()` method
2. Store R² in Optuna trial user_attrs
3. Report R² in results summary

### Medium-term

1. Add R² to all result files (JSON)
2. Display R² in tune summary output
3. Add R² to optuna_analyze_results.py

### Long-term

1. Add confusion matrix / residual analysis
2. Add per-sample prediction intervals
3. Cross-validation with multiple folds
4. Validation on held-out test set from different distribution

---

## Code References

**Current Calculation Locations:**

- MSE/RMSE: `trainer.py` lines 150-227 (train_epoch, validate, test methods)
- Denormalization: `optuna_trainable.py` lines 208-219
- Analysis: `optuna_analyze_results.py` display_summary()

**Dataset Statistics:**

- Stored in: `test_loader.dataset.target_std`, `test_loader.dataset.target_mean`
- Set during: `ChemicalDensityDataset.__init__()` in `data_loader.py`
