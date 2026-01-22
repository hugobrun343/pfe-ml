# Format de Sortie : Patches .nii.gz

## Structure des fichiers

Après le preprocessing, chaque patch est sauvegardé comme un fichier `.nii.gz` indépendant.

### Organisation des dossiers

```
output/preprocessed_{H}x{W}x{D}_{timestamp}/
├── train/
│   ├── stack_000001_patch_00_00.nii.gz
│   ├── stack_000001_patch_00_01.nii.gz
│   ├── stack_000001_patch_00_02.nii.gz
│   ├── ...
│   ├── stack_000001_patch_03_03.nii.gz  (dernier patch du volume 1)
│   ├── stack_000002_patch_00_00.nii.gz  (premier patch du volume 2)
│   └── ...
├── test/
│   └── ... (même structure)
├── patches_info.json    # Métadonnées complètes
└── metadata.json         # Configuration et statistiques
```

### Nommage des fichiers

Format : `{stack_id}_patch_{i:02d}_{j:02d}.nii.gz`

- `stack_id` : ID du volume d'origine (ex: `stack_000001`)
- `i` : Position ligne dans la grille (0 à n_patches_h-1)
- `j` : Position colonne dans la grille (0 à n_patches_w-1)

**Exemple pour grille 4×4 (16 patches par volume) :**
- `stack_000001_patch_00_00.nii.gz` → Patch en haut à gauche (i=0, j=0)
- `stack_000001_patch_00_03.nii.gz` → Patch en haut à droite (i=0, j=3)
- `stack_000001_patch_03_00.nii.gz` → Patch en bas à gauche (i=3, j=0)
- `stack_000001_patch_03_03.nii.gz` → Patch en bas à droite (i=3, j=3)

## Fichier `patches_info.json`

Contient toutes les métadonnées pour chaque patch.

### Structure

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
  {
    "filename": "stack_000001_patch_00_01.nii.gz",
    "split": "train",
    "stack_id": "stack_000001",
    "label": 0,
    "position_i": 0,
    "position_j": 1
  },
  ...
]
```

### Champs

- **`filename`** : Nom du fichier `.nii.gz`
- **`split`** : `"train"` ou `"test"`
- **`stack_id`** : ID du volume d'origine
- **`label`** : `0` (SAIN) ou `1` (MALADE)
- **`position_i`** : Ligne dans la grille (0-indexed)
- **`position_j`** : Colonne dans la grille (0-indexed)

### Utilisation

```python
import json

# Charger les métadonnées
with open('patches_info.json', 'r') as f:
    patches_info = json.load(f)

# Trouver tous les patches d'un volume
volume_patches = [p for p in patches_info if p['stack_id'] == 'stack_000001']

# Trouver tous les patches d'une position spécifique
corner_patches = [p for p in patches_info if p['position_i'] == 0 and p['position_j'] == 0]

# Filtrer par split
train_patches = [p for p in patches_info if p['split'] == 'train']
```

## Fichier `metadata.json`

Contient la configuration et les statistiques du preprocessing.

### Structure

```json
{
  "created": "2026-01-20T14:30:22.123456",
  "config": {
    "target_height": 256,
    "target_width": 256,
    "target_depth": 32,
    "n_patches_h": 4,
    "n_patches_w": 4,
    "n_patches_per_volume": 16,
    "slice_selection": "intensity",
    "normalization": "z-score"
  },
  "stats": {
    "train_volumes": 52,
    "test_volumes": 15,
    "train_patches": 832,
    "test_patches": 240,
    "errors": 0
  }
}
```

## Format des données dans les fichiers .nii.gz

### Dimensions

Chaque fichier `.nii.gz` contient un patch de dimensions :
- **Spatial** : `H × W` (ex: 256×256)
- **Profondeur** : `D` (ex: 32 slices)
- **Canaux** : `3` (RGB)

Format NIfTI interne : `(D, H, W, 3)` = `(Z, Y, X, C)`

### Type de données

- **Type** : `float32`
- **Normalisation** : Z-score (moyenne=0, std=1) par patch
- **Valeurs** : Normalisées (typiquement entre -3 et +3)

### Chargement

```python
import nibabel as nib
import numpy as np

# Charger un patch
img = nib.load('stack_000001_patch_00_00.nii.gz')
patch = img.get_fdata()  # Shape: (32, 256, 256, 3)

# Convertir pour PyTorch: (C, D, H, W)
patch_torch = np.transpose(patch, (3, 0, 1, 2))  # → (3, 32, 256, 256)
```

## Utilisation avec PyTorch

### Dataset

```python
from scripts.dataset_nii import NIIPatchDatasetWithLabels
import torch

# Charger le dataset
dataset = NIIPatchDatasetWithLabels(
    patches_dir="output/preprocessed_256x256x32_*/train/",
    patches_info_json="output/preprocessed_256x256x32_*/patches_info.json",
    split="train"
)

# DataLoader (multiprocessing-safe)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Pas de problème avec .nii.gz !
    shuffle=True
)

# Utilisation
for patch, label in dataloader:
    # patch: (batch, 3, 32, 256, 256)  [C, D, H, W]
    # label: (batch,)
    pass
```

### Récupérer les métadonnées

```python
# Obtenir les infos d'un patch
metadata = dataset.get_metadata(0)
print(f"Volume: {metadata['stack_id']}")
print(f"Position: ({metadata['position_i']}, {metadata['position_j']})")
```

## Avantages du format .nii.gz

1. **Multiprocessing-safe** : Chaque fichier est indépendant, pas de conflit entre workers
2. **Simple** : Un patch = un fichier, facile à déboguer
3. **Standard** : Format NIfTI standard en imagerie médicale
4. **Traçabilité** : `patches_info.json` contient toutes les infos (split, volume, position, label)
5. **Flexible** : Facile d'ajouter/supprimer des patches sans tout recalculer
6. **Compatible** : Compatible avec tous les outils d'imagerie médicale

## Exemple complet : Récupérer tous les patches d'un volume

```python
import json
from pathlib import Path
import nibabel as nib
import numpy as np

# Charger les métadonnées
with open('patches_info.json', 'r') as f:
    patches_info = json.load(f)

# Trouver tous les patches d'un volume
stack_id = 'stack_000001'
volume_patches = [p for p in patches_info if p['stack_id'] == stack_id]

# Charger tous les patches
patches = []
for patch_info in sorted(volume_patches, key=lambda x: (x['position_i'], x['position_j'])):
    patch_file = Path('train') / patch_info['filename']
    img = nib.load(patch_file)
    patch = img.get_fdata()  # (32, 256, 256, 3)
    patches.append(patch)

# patches est maintenant une liste de 16 patches (pour grille 4×4)
# Chaque patch: (32, 256, 256, 3)
```
