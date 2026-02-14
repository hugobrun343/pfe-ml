#!/bin/bash
#SBATCH --job-name=resume-convnext3d-large
#SBATCH --partition=gpu-dedicated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/brunh/scratch_brunh/pfe-ml/gpu-lightning/slurm_logs/%x_%j.out
#SBATCH --error=/home/brunh/scratch_brunh/pfe-ml/gpu-lightning/slurm_logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

ROOT="/home/brunh/scratch_brunh/pfe-ml/gpu-lightning"
LOG_DIR="$ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

source /storage/simple/users/brunh/miniforge3/etc/profile.d/conda.sh
conda activate py312

CONFIG="$ROOT/configs/train/train_convnext3d_large.yaml"
CKPT="/home/brunh/scratch_brunh/pfe-ml/_runs/train_convnext3d_large_adamw_lr1e4_20260213_085009/checkpoints/best_model.ckpt"
WANDB_ID="rqq008ax"

echo "========================================="
echo "  RESUME: ConvNeXt3D-Large single-split"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "WandB resume ID: $WANDB_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

nvidia-smi

export PYTHONUNBUFFERED=1
RUN_DIR="/home/brunh/scratch_brunh/pfe-ml/_runs/train_convnext3d_large_adamw_lr1e4_20260213_085009"

python3 -u "$ROOT/step4_train.py" \
    --config "$CONFIG" \
    --ckpt_path "$CKPT" \
    --wandb_resume_id "$WANDB_ID" \
    --run_dir "$RUN_DIR"

echo "End time: $(date)"
echo "Resume training complete!"
