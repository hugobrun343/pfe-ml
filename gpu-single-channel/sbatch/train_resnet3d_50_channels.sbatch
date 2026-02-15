#!/bin/bash
#SBATCH --job-name=sc-resnet3d-50
#SBATCH --array=0-2
#SBATCH --partition=gpu-dedicated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/brunh/scratch_brunh/pfe-ml/gpu-single-channel/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/brunh/scratch_brunh/pfe-ml/gpu-single-channel/slurm_logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

ROOT="/home/brunh/scratch_brunh/pfe-ml/gpu-single-channel"
LOG_DIR="$ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

source /storage/simple/users/brunh/miniforge3/etc/profile.d/conda.sh
conda activate py312

CONFIGS=(
  "$ROOT/configs/train/train_resnet3d_50_ch1.yaml"
  "$ROOT/configs/train/train_resnet3d_50_ch2.yaml"
  "$ROOT/configs/train/train_resnet3d_50_ch3.yaml"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$IDX]}"
CH=$((IDX + 1))

echo "========================================="
echo "  ResNet3D-50 -- Channel $CH"
echo "========================================="
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
echo "Channel $CH training complete!"
