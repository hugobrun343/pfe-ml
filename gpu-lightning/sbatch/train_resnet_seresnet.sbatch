#!/bin/bash
#SBATCH --job-name=gpu-lightning-train
#SBATCH --array=0-3
#SBATCH --partition=gpu-dedicated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_4g.40gb:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/brunh/scratch_brunh/pfe-ml/gpu-lightning/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/brunh/scratch_brunh/pfe-ml/gpu-lightning/slurm_logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

ROOT="/home/brunh/scratch_brunh/pfe-ml/gpu-lightning"
LOG_DIR="$ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

source /storage/simple/users/brunh/miniforge3/etc/profile.d/conda.sh
conda activate py312

CONFIGS=(
  "$ROOT/configs/train/train_resnet3d_50.yaml"
  "$ROOT/configs/train/train_resnet3d_101.yaml"
  "$ROOT/configs/train/train_seresnet3d_50.yaml"
  "$ROOT/configs/train/train_seresnet3d_101.yaml"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$IDX]}"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $IDX"
echo "Config: $CONFIG"
echo "Running on node(s): $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

nvidia-smi

export PYTHONUNBUFFERED=1
python3 -u "$ROOT/step4_train.py" --config "$CONFIG"

echo "End time: $(date)"
echo "Training complete!"
