#!/usr/bin/env bash
# Lance les runs train v1.7 à v1.10 pour split v1.1-s456
set -e
cd "$(dirname "$0")"
mkdir -p logs

for v in 1.7 1.8 1.9 1.10; do
  echo "=== Run train v${v} s456 ==="
  python train.py --config config/train_config_v${v}_s456.yaml 2>&1 | tee logs/train_v${v}_s456.log
done

echo "=== Done ==="
