# run-analytics

**Step 5 of the pipeline.** Post-training analysis tools for inspecting training run results, detecting data leakage, identifying problematic stacks, and comparing patch-level vs volume-level metrics.

## Purpose

Three independent analysis tools that operate on run directories produced by `gpu-lightning`:

1. **Split integrity check** — verifies that no stack or patch appears in both the train and validation sets of a run (detects data leakage).
2. **Problematic stack ranking** — across multiple runs, identifies stacks that are consistently misclassified (useful for dataset inspection and identifying hard cases).
3. **Aggregated F1 report** — compares patch-level F1 (per patch) to volume-level F1 (aggregated: mean probability per stack, threshold 0.5). Finds the best run by patch-level `f1_class_1`.

## When to use it

Run after `gpu-lightning` training completes, on one or more run directories. Particularly useful when comparing multiple runs (different seeds, preprocessing versions, or model architectures).

**Prerequisites:** one or more completed run directories from `gpu-lightning` (containing `results/` with prediction JSON files).

## Usage

All tools are run as Python modules from inside the `run-analytics/` directory:

```bash
cd run-analytics

# 1. Check that train/val sets are cleanly separated
python -m scripts.validate_split_integrity /path/to/_runs/train_resnet3d_50_*/

# 2. Find stacks that are consistently wrong across multiple runs
python -m scripts.rank_problematic_stacks \
    /path/to/_runs/train_resnet3d_50_*/ \
    /path/to/_runs/train_resnet3d_101_*/

# 3. Report patch-level vs volume-level F1 for the best run
python -m scripts.report_aggregated_f1 \
    /path/to/_runs/train_resnet3d_50_*/ \
    /path/to/_runs/train_resnet3d_101_*/
```

Multiple run directories can be passed as positional arguments for tools 2 and 3.

## Structure

```
run-analytics/
└── scripts/
    ├── __init__.py
    ├── io.py                       # resolve_run_paths() — resolve and validate run directory paths
    ├── extract.py                  # Extract predictions and labels from run result files
    ├── validate.py                 # Isolation check logic (stack-level and patch-level)
    ├── validate_split_integrity.py # Entrypoint: tool 1
    ├── rank_problematic_stacks.py  # Entrypoint: tool 2
    ├── stack_ranking.py            # Per-stack misclassification rate computation
    ├── aggregated_metrics.py       # Patch → stack aggregation, best run selection
    ├── report.py                   # Text report formatting helpers
    └── report_aggregated_f1.py     # Entrypoint: tool 3
```

## Tool details

### 1. validate_split_integrity

Checks for data leakage in a run directory. Verifies:
- No `stack_id` appears in both train and validation sets
- No patch filename appears in both sets

Exits with a non-zero code and prints the offending entries if any leakage is found.

### 2. rank_problematic_stacks

Across all provided runs, computes the misclassification rate for each stack (fraction of runs where the stack was predicted incorrectly). Outputs a ranked list of stacks from most to least frequently misclassified.

Useful to identify stacks that are inherently difficult (ambiguous label, poor acquisition quality, edge cases).

### 3. report_aggregated_f1

For each provided run, computes:
- **Patch-level F1:** standard F1 computed per individual patch
- **Volume-level F1:** each stack is predicted by averaging patch probabilities, then thresholding at 0.5

Selects the best run by patch-level `f1_class_1` (F1 for the MALADE class), then reports both metrics side by side to show the gap between patch and volume performance.
