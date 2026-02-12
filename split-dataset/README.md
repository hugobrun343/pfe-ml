# Dataset Split

Stratified train/test splits and cross-validation folds from the enriched dataset.

## Structure

```
split-dataset/
├── config/
│   ├── split_v1.1_config.yaml          # Simple split (full dataset, 20% test)
│   ├── split_test_config.yaml          # Filtered subset (DTA, M, D, stretch 2, P80)
│   └── cross_validation_config.yaml    # 5-fold CV + 10% hold-out test
├── scripts/
│   ├── split_utils/               # Shared library
│   │   ├── __init__.py
│   │   ├── io.py                  # load / save
│   │   ├── filters.py             # filtering + exclusion
│   │   ├── stratify.py            # stratified split + k-fold
│   │   ├── stats.py               # distribution statistics
│   │   ├── formatting.py          # text summary helpers
│   │   └── checks.py             # isolation + distribution checks
│   ├── create_train_test_split.py # Simple train/test
│   └── create_cv_split.py         # Cross-validation
└── README.md
```

## Simple split

```bash
cd split-dataset
python scripts/create_train_test_split.py \
    -c config/split_v1.1_config.yaml \
    -i /home/brunh/scratch_lirmm-ar/PFE_Fiorio/ds_shared/ds.json \
    -o /home/brunh/scratch_brunh/pfe-ml/_splits/v1.1/train_test_split.json
```

Output (timestamped `run_*/` subdir):
- `train_test_split.json` — train/test stack IDs
- `split_summary.txt` — statistics

## Cross-validation

```bash
cd split-dataset
python scripts/create_cv_split.py \
    -c config/cross_validation_config.yaml \
    -i /home/brunh/scratch_lirmm-ar/PFE_Fiorio/ds_shared/ds.json \
    -o /home/brunh/scratch_brunh/pfe-ml/_splits/cv-5fold-v1
```

Output (timestamped `run_*/` subdir):
- `train_test_split_fold_0.json` ... `_fold_4.json` — pipeline-compatible
  (same format as simple split: `"train"` + `"test"` keys)
- `cv_global.json` — hold-out test set + all fold assignments
- `cv_checks.txt` — isolation & distribution checks
- `cv_summary.txt` — per-fold statistics

## Configuration

All configs use:
- `exclude_stacks` — IDs to remove
- `filters` — age (min/max), axial_stretch, pressure, region, classe, genetic, sex, orientation
- `split` — test_size, random_seed, stratify_by, n_folds (CV only)
- `output` — generate_summary, generate_checks, indent_json
