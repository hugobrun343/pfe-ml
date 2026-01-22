#!/bin/bash

cd /mnt/pve/work/work-hugo/preprocess-dataset

# Lancer le preprocessing une seule fois (tous les volumes)
python scripts/preprocess_volumes_nii.py --config config/preprocess_config_256_split1.yaml
