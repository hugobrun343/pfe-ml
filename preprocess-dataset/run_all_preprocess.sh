#!/bin/bash
# Run preprocessing configs v1.7 through v1.10.

LOGS_DIR="/mnt/pve/work/work-hugo/preprocess-dataset/logs"
mkdir -p "$LOGS_DIR"

for v in 7 8 9 10; do
  python3 scripts/preprocess_volumes_nii.py --config "config/preprocess_config_v1.${v}.yaml" --workers 16 --check | tee "$LOGS_DIR/preprocess_v1.${v}.log"
done
