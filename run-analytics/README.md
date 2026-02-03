# run-analytics

Tools to analyze training runs.

## Scripts

### 1. Validate split integrity

Check that no stack or patch appears in both train and validation.

```bash
python -m scripts.validate_split_integrity <run_dir>
```

### 2. Rank problematic stacks

List stacks with worst prediction rate across multiple runs (useful to find consistently misclassified volumes).

```bash
python -m scripts.rank_problematic_stacks <run_dir> [run_dir ...]
```

### 3. Report aggregated (volume-level) F1

Find the best run (by patch-level f1_class_1) and report patch vs volume-level metrics. Volume-level = mean probability per stack, threshold 0.5.

```bash
python -m scripts.report_aggregated_f1 <run_dir> [run_dir ...]
```
