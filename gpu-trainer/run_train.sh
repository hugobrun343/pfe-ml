#!/bin/bash
# Run all 8 training configs (100 epochs each)

set -e  # Exit on first error

cd /mnt/pve/work/work-hugo/gpu-trainer

echo "=========================================="
echo "Running 8 full training runs (100 epochs)"
echo "=========================================="

# ResNet family
echo -e "\n[1/8] Training ResNet3D-50..."
python train.py --config config/train/train_resnet3d_50.yaml 2>&1 | tee logs/train_resnet3d_50.log

echo -e "\n[2/8] Training ResNet3D-101..."
python train.py --config config/train/train_resnet3d_101.yaml 2>&1 | tee logs/train_resnet3d_101.log

echo -e "\n[3/8] Training SE-ResNet3D-50..."
python train.py --config config/train/train_seresnet3d_50.yaml 2>&1 | tee logs/train_seresnet3d_50.log

echo -e "\n[4/8] Training SE-ResNet3D-101..."
python train.py --config config/train/train_seresnet3d_101.yaml 2>&1 | tee logs/train_seresnet3d_101.log

# ViT family
echo -e "\n[5/8] Training ViT3D-Base..."
python train.py --config config/train/train_vit3d_base.yaml 2>&1 | tee logs/train_vit3d_base.log

echo -e "\n[6/8] Training ViT3D-Large..."
python train.py --config config/train/train_vit3d_large.yaml 2>&1 | tee logs/train_vit3d_large.log

# ConvNeXt family
echo -e "\n[7/8] Training ConvNeXt3D-Small..."
python train.py --config config/train/train_convnext3d_small.yaml 2>&1 | tee logs/train_convnext3d_small.log

echo -e "\n[8/8] Training ConvNeXt3D-Large..."
python train.py --config config/train/train_convnext3d_large.yaml 2>&1 | tee logs/train_convnext3d_large.log

echo -e "\n=========================================="
echo "All 8 training runs completed!"
echo "=========================================="
