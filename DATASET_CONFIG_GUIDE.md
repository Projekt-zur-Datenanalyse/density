# Dataset Configuration Guide

## Overview

The data loader now supports **two dataset formats**:

1. **Original Format**: 4 separate CSV files with row-based structure (Dataset_1.csv, Dataset_2.csv, etc.)
2. **Unified Format**: Single cleaned CSV file with column-based structure (Dataset.csv)

### Quick Switch

To switch between formats, edit the top of `data_loader.py`:

```python
USE_UNIFIED_DATASET = False  # Set to True for unified dataset, False for separate files
```

---

## Dataset Comparison

### Original Format (4 Separate Files)

- **Files**: Dataset_1.csv, Dataset_2.csv, Dataset_3.csv, Dataset_4.csv
- **Format**: Row-based with first column as index
- **Total Samples**: 7,793
- **Density Range**: 0.00 - 939.21 kg/m³
- **Density Std Dev**: 201.44 kg/m³
- **Feature Ranges**:
  - SigC: [0.0500, 0.3500]
  - SigH: [0.0450, 0.3499]
  - EpsC: [0.1500, 1.1501]
  - EpsH: [0.0100, 0.1499]

### Unified Format (Dataset.csv)

- **File**: Dataset.csv (cleaned, unified data)
- **Format**: Column-based with headers (SigC, SigH, EpsC, EpsH, density)
- **Total Samples**: 6,492
- **Density Range**: 30.51 - 939.21 kg/m³
- **Density Std Dev**: 104.26 kg/m³
- **Feature Ranges**:
  - SigC: [0.0500, 0.3500]
  - SigH: [0.0451, 0.3476]
  - EpsC: [0.1510, 1.1501]
  - EpsH: [0.0101, 0.1499]

---

## Key Differences

| Aspect              | Original                   | Unified                       |
| ------------------- | -------------------------- | ----------------------------- |
| **Number of Files** | 4 separate files           | 1 combined file               |
| **Storage Format**  | Rows (features as rows)    | Columns (features as columns) |
| **Samples**         | 7,793                      | 6,492                         |
| **Density Range**   | 0.00 - 939.21              | 30.51 - 939.21                |
| **Std Dev**         | 201.44 kg/m³               | 104.26 kg/m³                  |
| **Data Quality**    | Mixed (includes anomalies) | Cleaned (anomalies removed)   |
| **Parsing**         | index_col=0 required       | Standard CSV format           |

---

## Important Notes

### RMSE Denormalization Scale

The RMSE denormalization scale (and all error metrics) automatically adjusts based on the dataset used:

```python
# Automatically determined from dataset
test_rmse_denorm = test_rmse_normalized * target_std

# Original Dataset:  test_rmse_denorm = test_rmse * 201.44
# Unified Dataset:   test_rmse_denorm = test_rmse * 104.26
```

### Feature Normalization

Features are normalized based on their min-max range in the dataset:

```python
normalized_feature = (feature - feature_mean) / feature_std
```

The normalization statistics are computed automatically when loading the dataset with `normalize_features=True`.

---

## How to Use

### Using Original Dataset (Default)

```python
from data_loader import ChemicalDensityDataLoader

loader = ChemicalDensityDataLoader(".")
train_loader, val_loader, test_loader, full_dataset = loader.load_dataset()
```

### Using Unified Dataset

**Option 1**: Edit the config at the top of data_loader.py:

```python
USE_UNIFIED_DATASET = True
```

**Option 2**: Temporarily override in code:

```python
import data_loader
data_loader.USE_UNIFIED_DATASET = True

loader = data_loader.ChemicalDensityDataLoader(".")
train_loader, val_loader, test_loader, full_dataset = loader.load_dataset()
```

---

## Output Information

When loading a dataset, the data loader prints:

- Dataset source (Original or Unified)
- Feature ranges for each input (SigC, SigH, EpsC, EpsH)
- Density range in kg/m³
- Density standard deviation (used for denormalization)
- Data split sizes (Train/Val/Test)

Example output:

```
[Unified Dataset] Loaded 6492 samples from Dataset.csv
Total samples: 6492
Features shape: (6492, 4)
Targets shape: (6492,)
Feature ranges:
  SigC: [0.0500, 0.3500]
  SigH: [0.0451, 0.3476]
  EpsC: [0.1510, 1.1501]
  EpsH: [0.0101, 0.1499]
Target (Density) range: [30.51, 939.21]
Target std dev: 104.26 kg/m³
Split: Train: 4675, Val: 1168, Test: 649
```

---

## Recommendation

**Use Unified Dataset (Dataset.csv) for**:

- Clean data without anomalies
- Faster training with fewer samples
- Lower variance in targets
- Better generalization

**Use Original Format (Dataset\_\*.csv) for**:

- Complete unfiltered data
- Larger training dataset
- Capturing anomalies and edge cases
- More robust models

---

## Implementation Details

### Original Format Loader (`_load_single_dataset`)

```python
df = pd.read_csv(dataset_path, index_col=0)
# Rows: SigC, SigH, EpsC, EpsH, density
# Columns: sample_1, sample_2, ...
```

### Unified Format Loader (`_load_unified_dataset`)

```python
df = pd.read_csv("Dataset.csv")
# Columns: SigC, SigH, EpsC, EpsH, density
# Rows: one sample per row
```

Both methods return:

- `features`: shape (n_samples, 4) - [SigC, SigH, EpsC, EpsH]
- `targets`: shape (n_samples,) - density values

---

## Troubleshooting

### Issue: "Dataset.csv not found"

- Make sure Dataset.csv is in the working directory
- Check that USE_UNIFIED_DATASET = True is set

### Issue: Data loading is slower with Unified Dataset

- The unified dataset has ~1,300 fewer samples but cleaner data
- This is expected behavior

### Issue: Different model performance with Unified vs Original

- This is expected due to different data distributions
- Original: std dev = 201.44, Unified: std dev = 104.26
- Models may need retraining when switching datasets

---

## Configuration Summary

```python
# At the top of data_loader.py:
USE_UNIFIED_DATASET = False  # <-- Change this to switch datasets
```

**Current setting**: `False` = Uses original 4-file format
