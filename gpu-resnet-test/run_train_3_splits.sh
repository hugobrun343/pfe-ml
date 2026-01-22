#!/bin/bash
# Script simple pour lancer 3 trainings avec 3 splits différents

cd /mnt/pve/work/work-hugo/gpu-resnet-test
DATE=$(date +%Y%m%d_%H%M%S)

echo "Lancement de 3 trainings avec différents splits"
echo "================================================"

python -u train.py --config config/config_split1.yaml \
    --resume checkpoints/resnet3d-50-full-dataset/latest_model.pth \
    2>&1 | tee "logs/train_split2_full_dataset_${DATE}.log"

python -u train.py --config config/config_split2.yaml \
    2>&1 | tee "logs/train_split3_full_dta_dataset_${DATE}.log"

echo "Tous les trainings sont terminés !"
