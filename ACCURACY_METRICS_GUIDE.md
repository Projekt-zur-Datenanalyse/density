# Understanding Accuracy Metrics in Optuna Analysis - CORRECTED

## CRITICAL UPDATE: Previous Analysis Was Wrong!

The denormalization was **not being applied correctly**. Here's the corrected analysis:

### Previous (WRONG) Values:
```
Test RMSE (denorm): 0.40 kg/m³   ← WRONG! This was normalized RMSE, not denormalized!
Test MAE (denorm): 0.15 kg/m³    ← WRONG! Same issue
Relative Accuracy: 99.80%         ← WRONG! Way too high
RMSE as % of Std Dev: 0.20%      ← WRONG! Way too low
```

### Correct Values:
```
Test RMSE (denorm): 80.04 kg/m³  ← CORRECT! (0.3973 × 201.44)
Test MAE (denorm): 29.71 kg/m³   ← CORRECT! (0.1475 × 201.44)
Relative Accuracy: 60.26%         ← CORRECT! (1 - 80.04/201.44)
RMSE as % of Std Dev: 39.74%     ← CORRECT! (80.04/201.44)
```

## What Happened

The `train_model()` function in `optuna_trainable.py` was returning the **normalized RMSE** for both the `test_rmse` and `test_rmse_denorm` parameters without actually denormalizing.

**The fix**: Multiply normalized error by `target_std` to get denormalized error:
```python
test_rmse_denorm = normalized_test_rmse * target_std
```

This applies the inverse normalization transformation:
- Normalization: `value_norm = (value - mean) / std`
- Denormalization: `value_denorm = value_norm * std + mean`
- For errors (which don't have a mean offset): `error_denorm = error_norm * std`

## Corrected Metrics Explained

### 1. **Test RMSE (denorm): 80.04 kg/m³**

- **Meaning**: Average prediction error is ±80.04 kg/m³
- **In context**: Dataset density values vary by ±201.44 kg/m³
- **Interpretation**: Model reduces uncertainty to ~40% of baseline variability
- **Is it good?**: Depends on your application - see benchmarks below

### 2. **Dataset Std Dev: 201.44 kg/m³**

- **Meaning**: Natural variability in density across all samples
- **Implication**: This is what the model must overcome to make good predictions
- **Baseline error**: Guessing the mean for every sample would give ~201.44 kg/m³ error

### 3. **RMSE as % of Std Dev: 39.74%**

- **Formula**: `(80.04 / 201.44) × 100% = 39.74%`
- **Meaning**: Model error is 39.74% of dataset variability
- **Benchmarks**:
  - < 10%: Excellent (>90% variance explained)
  - 10-20%: Very good (>80% variance explained)
  - 20-30%: Good (>70% variance explained)
  - 30-40%: **Acceptable (60-70% variance explained) ← Your model is here**
  - 40-50%: Fair (50-60% variance explained)
  - > 50%: Poor (<50% variance explained)

### 4. **Relative Accuracy: 60.26%**

- **Formula**: `(1 - 39.74%) × 100% = 60.26%`
- **Meaning**: Model explains 60.26% of the variance in the data
- **NOT**: 60% of individual predictions are correct (that's classification)
- **Actually**: 60% of the total unpredictability is removed by the model

## Is 60.26% Relative Accuracy Good?

**YES, for a regression model!** Here's why:

### ✅ Positive Aspects:

1. **Baseline comparison**:
   - Baseline: Predict mean always = 201.44 kg/m³ error
   - Your model: 80.04 kg/m³ error
   - **2.5x better than baseline!**

2. **Regression is hard**:
   - Classification (binary): 50% = random coin flip
   - Regression (continuous): 60% variance explained is solid
   - Predicting exact continuous values is inherently harder

3. **Practical performance**:
   - If true value is 600 kg/m³, your model predicts 600 ± 80 kg/m³
   - That's ±13.3% error, which is reasonable for many applications
   - Better than a 50/50 guess but not perfect

### ⚠️ When to Improve:

1. **Laboratory applications**: <20% error typically needed
2. **Industrial estimation**: 10-20% error often acceptable
3. **Research applications**: Depends on your tolerance
4. **Baseline comparison**: Compare against domain experts or competing methods

## Real-World Interpretation

| Scenario | True Value | Model Prediction | Error |
|----------|-----------|------------------|-------|
| Chemical A | 650 kg/m³ | 650 ± 80 | ±12.3% |
| Chemical B | 750 kg/m³ | 750 ± 80 | ±10.7% |
| Chemical C | 400 kg/m³ | 400 ± 80 | ±20.0% |

**Note**: Error varies by value since we're reporting absolute error, not relative error.

## Comparison with Other Metrics

```
Model Performance Tiers (Relative Accuracy):

Tier    Relative Acc    Error %    Interpretation
────────────────────────────────────────────────
S       > 95%          < 5%       Exceptional (rarely achievable)
A       85-95%         5-15%      Excellent (very good for regression)
B       70-85%         15-30%     Good (solid model)
C       50-70%         30-50%     Acceptable (your model) ← HERE
D       30-50%         50-70%     Fair (needs improvement)
F       < 30%          > 70%      Poor (much worse than baseline)
```

## What Went Wrong in the Previous Analysis

The bug was in `optuna_trainable.py`, line 225:

```python
# WRONG (old code):
return best_val_loss, test_rmse, test_rmse, mae
            ↑              ↑           ↑       ↑
      val loss        norm RMSE   norm RMSE (copied!)  norm MAE
                      (should be          ↑
                       denormalized)  Should multiply by target_std

# CORRECT (fixed code):
return best_val_loss, test_rmse, test_rmse_denorm, mae_denorm
                                      ↑ multiply by target_std
```

The second `test_rmse` parameter should have been denormalized but wasn't.

## How to Improve Model Performance

1. **More training data**: Collect more diverse samples
2. **Feature engineering**: Add domain-specific features if possible
3. **Hyperparameter optimization**: Continue using Optuna with more trials
4. **Try other architectures**: Test CNN or GNN models
5. **Ensemble methods**: Combine multiple models
6. **Data preprocessing**: Ensure proper normalization and outlier handling

## Key Takeaway

Your model's **60.26% relative accuracy** with **80.04 kg/m³ error** means:

✅ The model learns meaningful patterns  
✅ Predictions are 2.5x better than baseline (mean guessing)  
✅ Solid performance for a regression model  
⚠️ Room for improvement if high precision needed  
📊 Continue hyperparameter tuning to enhance further

## Summary Table

| Metric | Value | Meaning |
|--------|-------|---------|
| Test RMSE (normalized) | 0.3973 | Error on normalized scale |
| Test RMSE (denormalized) | 80.04 kg/m³ | Error in original units |
| Test MAE (denormalized) | 29.71 kg/m³ | Median error in original units |
| Dataset Std Dev | 201.44 kg/m³ | Baseline variability |
| Error as % of Std | 39.74% | How much variability remains |
| Relative Accuracy | 60.26% | Fraction of variance explained |

---

**Note**: Always verify denormalization is working correctly in production! The previous analysis showed how easy it is to miss this critical step.
