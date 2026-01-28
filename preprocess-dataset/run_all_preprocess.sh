#!/bin/bash
# Script to run all 6 preprocessing configurations.

LOGS_DIR="/mnt/pve/work/work-hugo/preprocess-dataset/logs"
mkdir -p "$LOGS_DIR"

python3 scripts/preprocess_volumes_nii.py --config config/preprocess_config_v1.1.yaml --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.1.log"
python3 scripts/preprocess_volumes_nii.py --config config/preprocess_config_v1.2.yaml --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.2.log"
python3 scripts/preprocess_volumes_nii.py --config config/preprocess_config_v1.3.yaml --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.3.log"
python3 scripts/preprocess_volumes_nii.py --config config/preprocess_config_v1.4.yaml --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.4.log"
python3 scripts/preprocess_volumes_nii.py --config config/preprocess_config_v1.5.yaml --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.5.log"
python3 scripts/preprocess_volumes_nii.py --config config/preprocess_config_v1.6.yaml --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.6.log"
