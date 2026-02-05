# GPU Trainer - 3D Deep Learning for Medical Image Classification

Training framework for 3D deep learning models on medical image patches (binary classification).

## Available Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| `resnet3d_50` | ~46M | ResNet3D-50 |
| `resnet3d_101` | ~85M | ResNet3D-101 |
| `seresnet3d_50` | ~49M | SE-ResNet3D-50 (Squeeze-and-Excitation) |
| `seresnet3d_101` | ~90M | SE-ResNet3D-101 |
| `vit3d_base` | ~86M | Vision Transformer 3D Base (12 layers) |
| `vit3d_large` | ~304M | Vision Transformer 3D Large (24 layers) |
| `convnext3d_small` | ~50M | ConvNeXt3D Small |
| `convnext3d_large` | ~198M | ConvNeXt3D Large |

## Project Structure

```
gpu-trainer/
├── train.py                    # Main training script
├── run_tests.sh                # Run all 8 models (10 epochs test)
├── run_train.sh                # Run all 8 models (100 epochs full)
├── config/
│   ├── test/                   # Test configs (10 epochs)
│   │   └── test_<model>.yaml
│   └── train/                  # Training configs (100 epochs)
│       └── train_<model>.yaml
├── scripts/
│   ├── models/                 # Model implementations
│   │   ├── resnet3d_50.py
│   │   ├── resnet3d_101.py
│   │   ├── seresnet3d_50.py
│   │   ├── seresnet3d_101.py
│   │   ├── vit3d_base.py
│   │   ├── vit3d_large.py
│   │   ├── convnext3d_small.py
│   │   └── convnext3d_large.py
│   ├── core/                   # Core modules (metrics)
│   └── helpers/                # Helper modules
│       ├── config_loader.py    # Config & args parsing
│       ├── run_setup.py        # Run directory setup
│       ├── data_loading.py     # Data loading
│       ├── model_setup.py      # Model & optimizer setup
│       ├── trainer.py          # Training loop
│       ├── results.py          # Results tracking
│       └── wandb_utils.py      # W&B logging
└── logs/                       # Training logs
```

## Usage

### Single Model Training

```bash
python train.py --config config/train/train_resnet3d_50.yaml
```

### Run All Models

```bash
# Test runs (10 epochs each)
./run_tests.sh

# Full training (100 epochs each)
./run_train.sh
```

### Resume Training

```bash
python train.py --config config/train/train_resnet3d_50.yaml \
    --resume ../_runs/<run_name>/checkpoints/latest_model.pth
```

## Configuration

Example config (`config/train/train_resnet3d_50.yaml`):

```yaml
run:
  run_name: train_resnet3d_50
  group: train_v2                # wandb group

data:
  preprocessed_dir: /mnt/pve/ds_shared/data/preprocess/preprocessed_256x256x32_v2
  train_test_split_json: /mnt/pve/work/work-hugo/_splits/full-dataset-v1.1-s42/train_test_split.json

input:
  batch_size: 64

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  num_workers: 12
  prefetch_factor: 2
  early_stopping_patience: 20
  early_stopping_min_delta: 0.0001

model:
  name: resnet3d_50              # Model selection
  in_channels: 3

system:
  device: cuda
```

### Model Selection

Set `model.name` to one of:
- `resnet3d_50`, `resnet3d_101`
- `seresnet3d_50`, `seresnet3d_101`
- `vit3d_base`, `vit3d_large`
- `convnext3d_small`, `convnext3d_large`

## Output Structure

Runs are saved to `../_runs/{run_name}/`:

```
_runs/{run_name}/
├── checkpoints/
│   ├── best_model.pth          # Best model (highest F1 Class 1)
│   └── latest_model.pth        # Latest model (for resuming)
├── results/
│   ├── training_results.json   # Training history
│   ├── training_metrics.png    # Loss & F1 plots
│   └── confusion_matrix.txt    # Final confusion matrix
├── data/
│   ├── config.yaml             # Copy of training config
│   ├── preprocessing_metadata.json
│   ├── train_test_split.json
│   └── wandb_info.json
└── wandb/                      # W&B logs
```

## Metrics

- **F1 Class 0**: F1-score for negative class (SAIN)
- **F1 Class 1**: F1-score for positive class (MALADE) - **primary metric**
- **Confusion Matrix**: TP, FP, TN, FN
- **Accuracy, Precision, Recall**

Early stopping and best model selection use **F1 Class 1**.

## Batch Sizes (RTX 4090 24GB)

Optimal batch sizes for each model:

| Model | Batch Size |
|-------|-----------|
| resnet3d_50 | 64 |
| resnet3d_101 | 48 |
| seresnet3d_50 | 56 |
| seresnet3d_101 | 40 |
| vit3d_base | 24 |
| vit3d_large | 8 |
| convnext3d_small | 32 |
| convnext3d_large | 16 |
