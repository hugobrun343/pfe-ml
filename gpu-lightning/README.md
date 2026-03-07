# gpu-lightning

**Step 3 of the pipeline — main training module.** Trains 3D binary classification models using PyTorch Lightning. Supports both simple train/val runs and 5-fold cross-validation with SLURM array jobs.

## Purpose

Binary classification of vascular tissue patches (SAIN=0, MALADE=1). Each patch is a `(3, 32, 256, 256)` tensor (channels × depth × height × width). The module is structured as numbered steps (2–5) for clarity.

**Primary metric:** `val_f1_mean = (F1_positive + F1_negative) / 2` — monitored for early stopping and checkpoint selection.

## When to use it

Run after `split-dataset`. Use the simple train configs for quick experiments, and the CV configs + SLURM scripts for final cross-validated results.

**Prerequisites:** preprocessed patches directory (from `preprocess-dataset`) + split JSON (from `split-dataset`) + W&B account.

## Structure

```
gpu-lightning/
├── step2_lightning_module.py    # LightningModule: Lit3DClassifier
├── step3_dataset.py             # Dataset and DataLoader: JSONSplitPatchDataset
├── step4_train.py               # Training entrypoint
├── step5_use_model.py           # Inference / checkpoint loading
├── config_utils.py              # YAML config loading and validation
├── run_utils.py                 # Run directory creation, metadata copy, checkpoint helpers
├── models/
│   ├── __init__.py              # get_model(name, in_channels, num_classes) factory
│   ├── resnet3d_50.py
│   ├── resnet3d_101.py
│   ├── seresnet3d_50.py
│   ├── seresnet3d_101.py
│   ├── vit3d_base.py
│   ├── vit3d_large.py
│   ├── convnext3d_small.py
│   ├── convnext3d_large.py
│   ├── densenet3d.py
│   └── swin3d.py
├── configs/
│   ├── train_config.yaml        # Base config template
│   ├── train/                   # Per-model train configs (100 epochs, full dataset split)
│   ├── test/                    # Per-model test configs (quick runs, few epochs)
│   └── cv/                      # Per-model × per-fold configs (5 folds × 6 models = 30 files)
└── sbatch/                      # SLURM job scripts for CV training on H100 cluster
    ├── cv_resnet3d_50.sbatch
    ├── cv_resnet3d_101.sbatch
    ├── cv_seresnet3d_50.sbatch
    ├── cv_seresnet3d_101.sbatch
    ├── cv_densenet3d_121.sbatch
    ├── cv_convnext3d_large.sbatch
    ├── cv_vit3d_base.sbatch
    └── resume_convnext3d_large_single.sbatch
```

## Usage

### Simple training run

```bash
cd gpu-lightning
python step4_train.py --config configs/train/train_resnet3d_50.yaml
```

### Resume a training run

```bash
python step4_train.py \
    --config configs/train/train_resnet3d_50.yaml \
    --ckpt_path /path/to/_runs/train_resnet3d_50_*/checkpoints/last.ckpt \
    --wandb_resume_id <wandb_run_id> \
    --run_dir /path/to/_runs/train_resnet3d_50_*/
```

### Cross-validation on SLURM cluster

```bash
# Launch 5 folds as a SLURM array job (H100 GPU, 8h, 16 CPUs, 128GB RAM)
sbatch sbatch/cv_resnet3d_50.sbatch
```

Each fold runs `step4_train.py` with its corresponding config from `configs/cv/`.

### Load a checkpoint for inference

```bash
python step5_use_model.py \
    --checkpoint /path/to/best_model.pth \
    --model-name resnet3d_50
```

## Configuration

```yaml
run:
  run_name: train_resnet3d_50
  runs_root: /path/to/_runs            # output directory for all run artefacts

data:
  preprocessed_dir: /path/to/patches   # directory with .npy patches
  train_test_split_json: /path/to/_splits/full-dataset-v1.1/train_test_split.json
  dataset_json: /path/to/dataset_final.json

input:
  batch_size: 32                        # adjust based on gpu-tester results

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam                       # adam | adamw
  weight_decay: 0.0
  scheduler: null                       # null | cosine (with optional warmup)
  warmup_epochs: 0
  gradient_clip_val: null
  limit_train_batches: 1.0
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 1.0e-05

model:
  name: resnet3d_50                     # see Models section below

wandb:
  project: resnet3d-binary-classification
  run_name: train_resnet3d_50
  group: train

system:
  accelerator: gpu
  devices: 1
  precision: 32-true                    # 32-true | 16-mixed | bf16-mixed
```

## Models

| Config name | Architecture |
|---|---|
| `resnet3d_50` | ResNet3D-50 |
| `resnet3d_101` | ResNet3D-101 |
| `seresnet3d_50` | SE-ResNet3D-50 |
| `seresnet3d_101` | SE-ResNet3D-101 |
| `vit3d_base` | ViT3D-Base (patch embedding + Transformer encoder) |
| `vit3d_large` | ViT3D-Large |
| `convnext3d_small` | ConvNeXt3D-Small |
| `convnext3d_large` | ConvNeXt3D-Large |
| `densenet3d_121` | DenseNet3D-121 |
| `swin3d_tiny` | Swin Transformer 3D Tiny |
| `swin3d_small` | Swin Transformer 3D Small |

All models take input `(B, 3, 32, 256, 256)` and output a single logit per sample (binary classification via `BCEWithLogitsLoss`).

## Outputs

Each run produces a timestamped directory under `runs_root`:

```
_runs/train_resnet3d_50_20260213_024142/
├── checkpoints/
│   └── best_model.pth           # Best checkpoint by val_f1_mean
├── results/
│   └── training_summary.json    # Best score, run name, checkpoint path
├── data/
│   ├── config.yaml              # Copy of the run config
│   ├── train_test_split.json    # Copy of the split used
│   └── wandb_info.json          # W&B project, run name, run ID and URL
├── analytics/
└── wandb/                       # W&B local logs
```

## Architecture: Lit3DClassifier

- **Loss:** `BCEWithLogitsLoss` (binary cross-entropy with logits)
- **Metrics logged:** `train_loss`, `val_loss`, `val_f1_pos`, `val_f1_neg`, `val_f1_mean`
- **Checkpoint selection:** maximize `val_f1_mean`
- **Optimizers:** Adam (default) or AdamW (`weight_decay` > 0 recommended with AdamW)
- **Scheduler:** optional cosine annealing with optional linear warmup ramp
