# Preprocessing

Transforme les volumes 3D en patches sauvegardés en `.nii.gz`.

## Pipeline

```
.nii.gz (1042×1042×[50-200]) → Découpe 8×8 → Sélection 32 slices → Resize 128×128×32 → .nii.gz
```

## Usage

```bash
python scripts/preprocess_volumes_nii.py --config config/preprocess_config_128.yaml
```

## Output

**Structure:**
```
output/preprocessed_{H}x{W}x{D}_{timestamp}/
├── train/
│   ├── stack_000001_patch_00_00.nii.gz
│   ├── stack_000001_patch_00_01.nii.gz
│   └── ...
├── test/
│   └── ...
├── patches_info.json    # Métadonnées: split, stack_id, label, position_i, position_j
└── metadata.json         # Config et statistiques
```

**Format `patches_info.json`:**
```json
[
  {
    "filename": "stack_000001_patch_00_00.nii.gz",
    "split": "train",
    "stack_id": "stack_000001",
    "label": 0,
    "position_i": 0,
    "position_j": 0
  },
  ...
]
```

**Exemple:** 52 volumes → 3,328 patches (52 × 64)

## Charger les données

```python
from scripts.dataset_nii import NIIPatchDataset
import torch

# Charger le dataset
dataset = NIIPatchDatasetWithLabels(
    patches_dir="output/preprocessed_128x128x32_*/train/",
    patches_info_json="output/preprocessed_128x128x32_*/patches_info.json",
    split="train"
)

# DataLoader PyTorch (multiprocessing-safe)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=32, 
    num_workers=4,
    shuffle=True
)

# Utilisation
for patch, label in dataloader:
    # patch: (batch, C, D, H, W)
    # label: (batch,)
    pass
```

## Config

```yaml
preprocessing:
  target_height/width/depth: 128×128×32
  n_patches_h/w: 8×8 = 64 patches
  slice_selection: intensity, variance, entropy, gradient
  normalization: z-score, min-max
```

## Dépendances

```bash
pip install numpy pyyaml scikit-image nibabel tqdm torch
```
