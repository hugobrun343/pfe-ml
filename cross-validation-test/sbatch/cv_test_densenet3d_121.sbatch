#!/bin/bash
#SBATCH --job-name=cv-test-densenet3d-121
#SBATCH --partition=gpu-dedicated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/brunh/scratch_brunh/pfe-ml/cross-validation-test/slurm_logs/%x_%j.out
#SBATCH --error=/home/brunh/scratch_brunh/pfe-ml/cross-validation-test/slurm_logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

ROOT="/home/brunh/scratch_brunh/pfe-ml/cross-validation-test"
LOG_DIR="$ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

source /storage/simple/users/brunh/miniforge3/etc/profile.d/conda.sh
conda activate py312

echo "========================================="
echo "  CV Test: DenseNet3D-121"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

nvidia-smi

export PYTHONUNBUFFERED=1
cd "$ROOT"
python3 -u run_cv_test.py --config configs/test_cv_densenet3d_121.yaml

echo "End time: $(date)"
echo "CV test complete!"
