#!/bin/bash
# Run 7 remaining test configs (10 epochs each)
# SE-ResNet3D-101 already completed

set -e  # Exit on first error

cd /mnt/pve/work/work-hugo/gpu-trainer

echo "=========================================="
echo "Running 7 model tests (10 epochs each)"
echo "=========================================="

# ResNet family
echo -e "\n[1/7] Testing ResNet3D-50..."
python train.py --config config/test/test_resnet3d_50.yaml 2>&1 | tee logs/test_resnet3d_50.log

echo -e "\n[2/7] Testing ResNet3D-101..."
python train.py --config config/test/test_resnet3d_101.yaml 2>&1 | tee logs/test_resnet3d_101.log

echo -e "\n[3/7] Testing SE-ResNet3D-50..."
python train.py --config config/test/test_seresnet3d_50.yaml 2>&1 | tee logs/test_seresnet3d_50.log

# SE-ResNet3D-101 SKIPPED (already completed)

# ViT family
echo -e "\n[4/7] Testing ViT3D-Base..."
python train.py --config config/test/test_vit3d_base.yaml 2>&1 | tee logs/test_vit3d_base.log

echo -e "\n[5/7] Testing ViT3D-Large..."
python train.py --config config/test/test_vit3d_large.yaml 2>&1 | tee logs/test_vit3d_large.log

# ConvNeXt family
echo -e "\n[6/7] Testing ConvNeXt3D-Small..."
python train.py --config config/test/test_convnext3d_small.yaml 2>&1 | tee logs/test_convnext3d_small.log

echo -e "\n[7/7] Testing ConvNeXt3D-Large..."
python train.py --config config/test/test_convnext3d_large.yaml 2>&1 | tee logs/test_convnext3d_large.log

echo -e "\n=========================================="
echo "All 7 tests completed!"
echo "=========================================="
