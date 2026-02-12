#!/bin/bash
#SBATCH --job-name=gpu-lightning-train
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/pve/work/work-hugo/gpu-lightning/slurm_logs/%x_%A_%a.out
#SBATCH --error=/mnt/pve/work/work-hugo/gpu-lightning/slurm_logs/%x_%A_%a.err
##SBATCH --partition=gpu
##SBATCH --account=<your_account>

set -euo pipefail

ROOT="/mnt/pve/work/work-hugo/gpu-lightning"
LOG_DIR="$ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "$ROOT/configs/train/train_resnet3d_50.yaml"
  "$ROOT/configs/train/train_resnet3d_101.yaml"
  "$ROOT/configs/train/train_seresnet3d_50.yaml"
  "$ROOT/configs/train/train_seresnet3d_101.yaml"
  "$ROOT/configs/train/train_convnext3d_small.yaml"
  "$ROOT/configs/train/train_convnext3d_large.yaml"
  "$ROOT/configs/train/train_vit3d_base.yaml"
  "$ROOT/configs/train/train_vit3d_large.yaml"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$IDX]}"

echo "[$(date)] job_id=${SLURM_JOB_ID:-na} array_task=${IDX}"
echo "[$(date)] config=${CONFIG}"

source /mnt/pve/miniforge3/etc/profile.d/conda.sh
conda activate py312

python "$ROOT/step4_train.py" --config "$CONFIG"

echo "[$(date)] done config=${CONFIG}"
