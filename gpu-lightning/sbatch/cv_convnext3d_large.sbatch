#!/bin/bash
#SBATCH --job-name=cv-convnext3d-large
#SBATCH --array=0-4
#SBATCH --partition=gpu-dedicated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=08:00:00
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
  "$ROOT/configs/cv/convnext3d_large_fold_0.yaml"
  "$ROOT/configs/cv/convnext3d_large_fold_1.yaml"
  "$ROOT/configs/cv/convnext3d_large_fold_2.yaml"
  "$ROOT/configs/cv/convnext3d_large_fold_3.yaml"
  "$ROOT/configs/cv/convnext3d_large_fold_4.yaml"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$IDX]}"

echo "========================================="
echo "  Cross-Validation: ConvNeXt3D-Large  Fold $IDX"
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
echo "Fold $IDX training complete!"
