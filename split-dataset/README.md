# split-dataset

**Step 2 of the pipeline.** Creates stratified train/val/test splits or 5-fold cross-validation folds from the enriched dataset JSON.

## Purpose

Splits are performed at the **stack level** (not patch level). Patch filtering is handled at load time by the training modules. All splits use iterative stratification to preserve class balance and metadata distributions (age, sex, region, pressure, axial stretch, orientation, genetic background) across train, val, and test sets.

## When to use it

Run after `preprocess-dataset` and before `gpu-lightning` or `gpu-single-channel`.

**Prerequisites:** enriched dataset JSON (`_dataset-json/dataset_final.json`).

## Structure

```
split-dataset/
├── config/
│   ├── split_v1.1_config.yaml          # Simple split: full dataset, 80/20, seed=123
│   ├── split_test_config.yaml          # Filtered subset (specific region/pressure/stretch)
│   └── cross_validation_config.yaml    # 5-fold CV + 10% hold-out test set
└── scripts/
    ├── create_train_test_split.py      # Entrypoint: simple train/test split
    ├── create_cv_split.py              # Entrypoint: cross-validation split
    └── split_utils/
        ├── io.py                       # Load/save dataset JSON and split JSON
        ├── filters.py                  # Stack filtering and exclusion
        ├── stratify.py                 # Stratified split + k-fold (iterative-stratification)
        ├── stats.py                    # Distribution statistics per split
        ├── formatting.py               # Text summary helpers
        └── checks.py                   # Isolation and distribution checks
```

## Simple train/test split

```bash
cd split-dataset
python scripts/create_train_test_split.py \
    -c config/split_v1.1_config.yaml \
    -i /path/to/dataset_final.json \
    -o /path/to/_splits/full-dataset-v1.1/train_test_split.json
```

**Output:**

| File | Description |
|---|---|
| `train_test_split.json` | `{"train": [stack_id, ...], "test": [stack_id, ...]}` |
| `split_summary.txt` | Per-split statistics (class balance, metadata distributions) |

## Cross-validation split

```bash
cd split-dataset
python scripts/create_cv_split.py \
    -c config/cross_validation_config.yaml \
    -i /path/to/dataset_final.json \
    -o /path/to/_splits/cv-5fold-v1
```

**Output:**

| File | Description |
|---|---|
| `train_test_split_fold_0.json` … `_fold_4.json` | Per-fold train/val splits (same format as simple split) |
| `cv_global.json` | Hold-out test set + all fold assignments |
| `cv_checks.txt` | Isolation and distribution validation |
| `cv_summary.txt` | Per-fold statistics |
| `plots_train/`, `plots_val/`, `plots_holdout/` | Distribution bar charts per feature |

The per-fold JSONs use the same `{"train": [...], "test": [...]}` format as the simple split, making them directly compatible with `gpu-lightning`'s training configs.

## Configuration

```yaml
exclude_stacks:
  - stack_000163    # stacks removed from all splits (e.g. acquisition issues)
  - stack_000147

filters:
  age: {min: 6, max: null}   # keep only animals >= 6 weeks
  sex: null                   # no filter (null = keep all)
  region: null
  # ... same pattern for: axial_stretch, pressure, classe, genetic, orientation

split:
  test_size: 0.2              # fraction for test set (simple split)
  # or:
  test_size: 0.1              # fraction for hold-out (CV split)
  n_folds: 5                  # number of CV folds
  random_seed: 123
  stratify_by:
    - Age
    - Sex
    - Region
    - Axial stretch
    - Pressure
    - Classe
    - Orientation
    - Genetic

output:
  generate_summary: true
  generate_checks: true
  indent_json: 2
```

## Tracked splits

Pre-generated splits are stored at the root of the repo in `_splits/`:

| Directory | Description |
|---|---|
| `_splits/full-dataset-v1.1/` | Simple 80/20 split, seed=123 |
| `_splits/full-dataset-v1.1-s42/` | Same config, seed=42 |
| `_splits/full-dataset-v1.1-s456/` | Same config, seed=456 |
| `_splits/full-dataset-v1.1-s789/` | Same config, seed=789 |
| `_splits/cv-5fold-v1/` | 5-fold CV + 10% hold-out, seed=123 |
