# pfe-ml

Binary classification of vascular tissue from two-photon microscopy 3D volumes.

The dataset consists of 771 NIfTI stacks (3-channel, ~1042×1042×50-200 voxels) of mouse/rat blood vessel segments acquired under various experimental conditions (pressure, axial stretch, orientation). The goal is to classify each volume as **SAIN** (healthy, label 0) or **MALADE** (diseased, label 1).

---

## Pipeline overview

The project is split into independent modules. Run them in this order:

```
Step 0  refacto-dataset        Build the enriched dataset JSON from raw TSV + folder structure
  │
  ├──►  voxel-analytics        (optional) Analyze voxel intensity distribution to guide normalization
  │
Step 1  preprocess-dataset     Extract 3D patches from NIfTI volumes (256×256×32, 3 channels)
  │
Step 2  split-dataset          Create stratified train/val/test splits or 5-fold CV folds
  │
  ├──►  gpu-tester             (optional) Profile VRAM usage to find max batch sizes
  │
Step 3  gpu-lightning          Train 3D classification models with PyTorch Lightning (main approach)
  │
  ├──►  gpu-single-channel     (optional) Channel ablation: train with 1 or 2 channels instead of 3
  │
Step 4  cross-validation-test  Run inference on all CV fold checkpoints, ensemble, report metrics
  │
Step 5  run-analytics          Post-training analysis: split integrity, stack ranking, aggregated F1
```

---

## Modules

| Module | Step | Description |
|---|---|---|
| [`refacto-dataset/`](refacto-dataset/) | 0 | Parse raw TSV database → enriched dataset JSON |
| [`voxel-analytics/`](voxel-analytics/) | optional | Voxel intensity statistics across all volumes |
| [`preprocess-dataset/`](preprocess-dataset/) | 1 | NIfTI → 3D patch extraction and normalization |
| [`split-dataset/`](split-dataset/) | 2 | Stratified train/test splits and k-fold CV |
| [`gpu-tester/`](gpu-tester/) | optional | VRAM grid search to determine viable batch sizes |
| [`gpu-lightning/`](gpu-lightning/) | 3 | PyTorch Lightning training (main, cross-validation) |
| [`gpu-single-channel/`](gpu-single-channel/) | optional | Channel ablation variant of gpu-lightning |
| [`cross-validation-test/`](cross-validation-test/) | 4 | CV inference, fold ensembling, volume-level metrics |
| [`run-analytics/`](run-analytics/) | 5 | Split integrity checks, problematic stack ranking, F1 reporting |

---

## Root artefacts

The following directories at the root of the repo contain tracked data artefacts produced by the pipeline:

| Directory | Produced by | Contents |
|---|---|---|
| `_dataset-json/` | `refacto-dataset` | Enriched dataset JSONs (`dataset_final.json`, backup) |
| `_dataset-tsv/` | source data | Original and final TSV database files |
| `_splits/` | `split-dataset` | Train/test split JSONs and CV fold JSONs with stratification plots |
| `_results/` | `cross-validation-test` | CV test results per model (`results.json`, `summary.txt`) |

Raw data (NIfTI volumes, preprocessed NPY patches) is **not tracked in git** — paths to these are configured per-machine in each module's YAML configs.

---

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate py312
```

The environment requires CUDA 12 and was tested on an H100 80GB (training) and L40S 48GB (VRAM profiling). Key dependencies: `torch 2.9`, `lightning 2.6`, `nibabel 5.3`, `monai 1.5`, `wandb 0.23`.

---

## Models

The following 3D architectures are implemented across `gpu-lightning`, `gpu-single-channel`, and `cross-validation-test`:

| Model | Description |
|---|---|
| ResNet3D-50 / 101 | 3D ResNet with bottleneck blocks |
| SE-ResNet3D-50 / 101 | ResNet3D with Squeeze-and-Excitation channels |
| ViT3D-Base / Large | 3D Vision Transformer |
| ConvNeXt3D-Small / Large | 3D ConvNeXt |
| DenseNet3D-121 | 3D DenseNet |
| Swin3D-Tiny / Small | 3D Swin Transformer |

Input: `(B, 3, 32, 256, 256)` — batch × channels × depth × height × width.

---

## Experiment tracking

All training runs log to [Weights & Biases](https://wandb.ai). Configure your W&B API key before running training:

```bash
wandb login
```
