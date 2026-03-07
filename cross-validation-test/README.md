# cross-validation-test

**Step 4 of the pipeline.** Runs inference on all 5 CV fold checkpoints, aggregates patch-level predictions to stack-level scores, ensembles the 5 models, and reports volume-level metrics on the hold-out test set.

## Purpose

After `gpu-lightning` trains 5 fold models for a given architecture, this module:

1. Loads the hold-out test patches (stacks excluded from all CV folds, defined in `cv_global.json`)
2. Runs inference with each of the 5 fold checkpoints
3. Aggregates patch scores → stack scores (mean probability per stack)
4. Ensembles the 5 models (mean of stack scores across folds)
5. Computes volume-level metrics (accuracy, F1, AUC) and saves a full results report

This gives a more reliable estimate of real-world performance than patch-level metrics, since a single volume's prediction is based on all its patches combined.

## When to use it

Run after `gpu-lightning` CV training is complete (all 5 folds trained for a given model).

**Prerequisites:**
- 5 fold checkpoints (`best_model.pth`) from `gpu-lightning` CV training
- `cv_global.json` from `split-dataset` (hold-out set definition)
- Preprocessed patches directory (from `preprocess-dataset`)

## Usage

```bash
cd cross-validation-test

# Test ResNet3D-50 (uses 5 fold checkpoints)
python run_cv_test.py --config configs/test_cv_resnet3d_50.yaml

# Test on SLURM cluster
sbatch sbatch/cv_test_resnet3d_50.sbatch
```

Available model configs: `resnet3d_50`, `resnet3d_101`, `seresnet3d_50`, `seresnet3d_101`, `densenet3d_121`, `convnext3d_large`.

## Configuration

```yaml
model:
  name: resnet3d_50

paths:
  preprocessed_dir: /path/to/patches          # .npy patches directory
  cv_global_json: /path/to/_splits/cv-5fold-v1/cv_global.json

checkpoints:
  - /path/to/_runs/cv_resnet3d_50_fold_0_*/checkpoints/best_model.pth
  - /path/to/_runs/cv_resnet3d_50_fold_1_*/checkpoints/best_model.pth
  - /path/to/_runs/cv_resnet3d_50_fold_2_*/checkpoints/best_model.pth
  - /path/to/_runs/cv_resnet3d_50_fold_3_*/checkpoints/best_model.pth
  - /path/to/_runs/cv_resnet3d_50_fold_4_*/checkpoints/best_model.pth

inference:
  batch_size: 32
  device: cuda

output:
  results_dir: /path/to/_results/cv_test_resnet3d_50
```

## Outputs

Results are saved to `results_dir` (pre-generated outputs tracked in `_results/`):

| File | Description |
|---|---|
| `results.json` | Full results: patch scores, stack scores, per-fold metrics, ensemble metrics |
| `summary.txt` | Human-readable summary: ensemble F1, AUC, accuracy, per-fold comparison |

**`results.json` structure:**

```json
{
  "config": { ... },
  "patch_scores": { "fold_0": {"patch_filename": prob, ...}, ... },
  "stack_scores": {
    "per_model": { "fold_0": {"stack_id": mean_prob, ...}, ... },
    "ensemble":  { "stack_id": mean_prob_across_folds, ... }
  },
  "labels": { "stack_id": 0_or_1, ... },
  "predictions": { "stack_id": 0_or_1, ... },
  "metrics": {
    "per_model": { "fold_0": {"f1_mean": ..., "auc": ..., ...}, ... },
    "ensemble":  { "f1_mean": ..., "auc": ..., "accuracy": ... }
  }
}
```

## Structure

```
cross-validation-test/
├── run_cv_test.py               # Main entrypoint (6-step pipeline)
├── lightning_module.py          # Lit3DClassifier (same as gpu-lightning, for checkpoint loading)
├── models/                      # Same model definitions as gpu-lightning
├── cv_test/
│   ├── config.py                # Config loading and validation
│   ├── dataset.py               # make_test_loader() — load hold-out patches from cv_global.json
│   ├── inference.py             # run_inference() — forward pass on all patches for one checkpoint
│   ├── aggregate.py             # aggregate_patches_to_stacks(), ensemble_stacks()
│   ├── metrics.py               # compute_metrics() — accuracy, F1, AUC at stack level
│   └── report.py                # save_results_json(), save_summary_txt()
├── configs/                     # Per-model test configs
└── sbatch/                      # SLURM job scripts
```

## Pipeline steps

```
[1/6] Load config
[2/6] Load hold-out test patches (from cv_global.json)
[3/6] Run inference with each of the 5 fold checkpoints → patch probabilities
[4/6] Aggregate patch scores → stack scores (mean prob per stack, per fold)
[5/6] Ensemble: average stack scores across all 5 folds → final prediction
[6/6] Compute metrics + save results.json and summary.txt
```

## Tracked results

Pre-computed results for 6 models are tracked at `_results/`:

| Model | Ensemble F1 | Notes |
|---|---|---|
| `cv_test_resnet3d_50` | see `_results/cv_test_resnet3d_50/summary.txt` | |
| `cv_test_resnet3d_101` | see summary | |
| `cv_test_seresnet3d_50` | see summary | |
| `cv_test_seresnet3d_101` | see summary | |
| `cv_test_densenet3d_121` | see summary | |
| `cv_test_convnext3d_large` | see summary | |
