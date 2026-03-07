# gpu-single-channel

**Optional ablation experiment.** Variant of `gpu-lightning` that adds a `channels` parameter to test whether using 1 or 2 input channels instead of all 3 affects classification performance.

## Purpose

The dataset volumes have 3 channels (three acquisition modes from two-photon microscopy). This module tests whether a model trained on a single channel can match the performance of a 3-channel model, which would simplify acquisition requirements.

Three models are tested across three channel counts (1, 2, 3): **ResNet3D-50**, **ResNet3D-101**, **DenseNet3D-121** → 9 total configurations.

## When to use it

Run in parallel with or after `gpu-lightning`. Requires the same preprocessed patches (3-channel `.npy` files) and the same split JSON. The dataset class selects the specified channels at load time.

**Prerequisites:** preprocessed patches (from `preprocess-dataset`) + split JSON (from `split-dataset`).

## Differences from gpu-lightning

| Feature | gpu-lightning | gpu-single-channel |
|---|---|---|
| Input channels | Always 3 | Configurable: 1, 2, or 3 |
| `input.channels` config key | Not present | Required |
| Dataset | Loads all channels | Selects first N channels |
| Model init | `in_channels=3` | `in_channels=cfg.channels` |
| Scope | All models, CV | 3 models, simple splits only |

## Structure

```
gpu-single-channel/
├── step2_lightning_module.py    # Same as gpu-lightning (Lit3DClassifier, in_channels param)
├── step3_dataset.py             # JSONSplitPatchDataset with channels selection
├── step4_train.py               # Same as gpu-lightning, passes channels=cfg.channels
├── step5_use_model.py           # Same as gpu-lightning
├── config_utils.py
├── run_utils.py
├── models/                      # Same architectures as gpu-lightning
└── configs/
    ├── train_config.yaml        # Base template (adds input.channels)
    └── train/
        ├── train_resnet3d_50_ch1.yaml    # ResNet3D-50, 1 channel
        ├── train_resnet3d_50_ch2.yaml    # ResNet3D-50, 2 channels
        ├── train_resnet3d_50_ch3.yaml    # ResNet3D-50, 3 channels (baseline)
        ├── train_resnet3d_101_ch1.yaml
        ├── train_resnet3d_101_ch2.yaml
        ├── train_resnet3d_101_ch3.yaml
        ├── train_densenet3d_121_ch1.yaml
        ├── train_densenet3d_121_ch2.yaml
        └── train_densenet3d_121_ch3.yaml
├── sbatch/
    ├── train_resnet3d_50_channels.sbatch
    ├── train_resnet3d_101_channels.sbatch
    └── train_densenet3d_121_channels.sbatch
```

## Usage

```bash
cd gpu-single-channel

# Train ResNet3D-50 with 1 channel
python step4_train.py --config configs/train/train_resnet3d_50_ch1.yaml

# Train ResNet3D-50 with 3 channels (3-channel baseline for comparison)
python step4_train.py --config configs/train/train_resnet3d_50_ch3.yaml

# Launch on SLURM (runs ch1, ch2, ch3 sequentially in a single job)
sbatch sbatch/train_resnet3d_50_channels.sbatch
```

## Configuration

Same structure as `gpu-lightning` plus the `input.channels` key:

```yaml
input:
  batch_size: 64
  channels: 1         # 1 | 2 | 3 — number of channels to use (first N channels selected)
```

All other configuration keys (`run`, `data`, `training`, `model`, `wandb`, `system`) are identical to `gpu-lightning`. See `gpu-lightning/README.md` for the full reference.

## Outputs

Same run directory structure as `gpu-lightning`, under `runs_root`. Run names include the channel count (e.g., `train_resnet3d_50_ch1_20260213_*`).
