# ResNet3D-50 Training

Training script for ResNet3D-50 binary classification on 3D medical image patches.

## Structure

```
gpu-resnet/
├── train.py              # Main training script
├── config/
│   └── config.yaml      # Training configuration
├── scripts/              # All code modules
│   ├── core/            # Core modules
│   │   ├── model.py     # ResNet3D-50 model definition
│   │   └── metrics.py   # Classification metrics (F1 per class, confusion matrix)
│   ├── helpers/         # Helper modules
│   │   ├── config_loader.py  # Configuration loading and argument parsing
│   │   ├── run_setup.py      # Run directory creation and metadata copying
│   │   ├── data_loading.py   # Data loading utilities
│   │   ├── model_setup.py    # Model and optimizer setup
│   │   ├── resume.py          # Checkpoint resuming utilities
│   │   ├── display.py         # Display utilities for results
│   │   ├── data_loader.py     # DataLoader creation
│   │   ├── dataset.py         # NIIPatchDataset for .nii.gz patches
│   │   ├── trainer.py         # Training loop with early stopping
│   │   ├── results.py         # Results tracking and JSON export
│   │   └── wandb_utils.py     # Weights & Biases logging
│   └── analytics/       # Results analysis scripts
# Runs are saved to _runs/ at work-hugo root (not in gpu-resnet/)
# Structure: ../_runs/{run_name}/
```

## Usage

```bash
python train.py --config config/config.yaml --run-name my_experiment
```

### Arguments

- `--config` (required): Path to YAML configuration file
- `--run-name` (required): Name for this training run (creates `_runs/{run_name}/` at work-hugo root)
- `--preprocessed-dir`: Override preprocessed directory from config
- `--train-test-split-json`: Override train/test split JSON from config
- `--batch-size`, `--epochs`, `--lr`, etc.: Override training parameters from config
- `--resume`: Path to checkpoint to resume from (e.g., `_runs/{run_name}/checkpoints/latest_model.pth`)
  - **Note**: The resume functionality uses factory functions to automatically extract hyperparameters from the checkpoint, so you don't need to match the original config

## Configuration

Edit `config/config.yaml` to configure:
- **Data paths**: `preprocessed_dir`, `train_test_split_json`
- **Model**: `in_channels`, `num_classes`
- **Training**: `batch_size`, `epochs`, `learning_rate`, `weight_decay`
- **System**: `device`, `num_workers`, `prefetch_factor`
- **Early stopping**: `early_stopping_patience`, `early_stopping_min_delta`

## Loading Models and Optimizers from Checkpoints

You can easily load models and optimizers from checkpoints using factory functions:

```python
from scripts.helpers.artefacts import load_model_from_checkpoint, load_optimizer_from_checkpoint
import torch

# Load model automatically (extracts hyperparameters from checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_from_checkpoint(
    Path('_runs/my_run_20260122_143052/checkpoints/best_model.pth'),
    device=device
)

# Load optimizer for the model
optimizer = load_optimizer_from_checkpoint(
    Path('_runs/my_run_20260122_143052/checkpoints/best_model.pth'),
    model=model
)

# Model and optimizer are ready to use!
```

You can also recreate a DataLoader from a config file:

```python
from scripts.helpers.artefacts import load_dataloader_from_config

# Load DataLoader from config
train_loader = load_dataloader_from_config(
    Path('config/config.yaml'),
    split='train',
    batch_size=32  # Optional: override batch size
)
```

## Output Structure

Each training run creates a directory `_runs/{run_name}/` at work-hugo root with:

- **`checkpoints/`**: Model checkpoints
  - `best_model.pth`: Best model (highest F1 Class 1)
  - `latest_model.pth`: Latest model (for resuming)
  
- **`results/`**: Training results
  - `training_results.json`: Complete training history with per-epoch metrics
  - `training_metrics.png`: Plot of loss and F1 scores over epochs
  - `confusion_matrix.txt`: Final confusion matrix and F1 scores
  
- **`analytics/`**: Analysis outputs (auto-generated)
  - `position_analysis_heatmap.png`: Success rate by patch position
  - `position_analysis_barchart.png`: Bar chart of position performance
  - `volume_analysis.png`: Success rate by volume
  - `analysis_report.txt`: Detailed text report
  
- **`data/`**: Data metadata (copied automatically)
  - `preprocessing_metadata.json`: Copy of preprocessing metadata (version, config, stats)
  - `train_test_split.json`: Copy of train/test split used for this run
  - `original_dataset.json`: Copy of original dataset JSON (for raw data traceability)
  - `wandb_info.json`: Wandb run information (run_id, url, run_name, project)
  
- **`wandb/`**: Weights & Biases logs

## Metrics

The training tracks:
- **F1 Class 0**: F1-score for negative class (SAIN)
- **F1 Class 1**: F1-score for positive class (MALADE)
- **Confusion Matrix**: TP, FP, TN, FN
- **Accuracy, Precision, Recall**: Standard classification metrics

Early stopping and best model selection use **F1 Class 1** as the primary metric.

## Resuming Training

```bash
python train.py --config config/config.yaml --run-name my_experiment \
    --resume ../_runs/my_experiment/checkpoints/latest_model.pth
```

The script will:
- **Automatically extract hyperparameters from the checkpoint** (using factory functions)
- Load model and optimizer state
- Continue from the next epoch
- Append new epochs to existing `training_results.json`
- Preserve all previous training history

## Analytics

After training completes, analytics are automatically generated. You can also run manually:

```bash
python scripts/analytics/analyze_results.py ../_runs/{run_name}/results/training_results.json \
    --output-dir ../_runs/{run_name}/analytics
```
