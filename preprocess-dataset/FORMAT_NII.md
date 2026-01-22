# Format de Sortie : Patches .nii.gz

## Structure des fichiers

Après le preprocessing, chaque patch est sauvegardé comme un fichier `.nii.gz` indépendant.

### Organisation des dossiers

```
output/preprocessed_{H}x{W}x{D}_{version}/
├── patches/
│   ├── stack_000001_patch_00_00.nii.gz
│   ├── stack_000001_patch_00_01.nii.gz
│   ├── ...
│   ├── stack_000001_patch_03_03.nii.gz
│   ├── stack_000002_patch_00_00.nii.gz
│   └── ...
├── patches_info.json    # Métadonnées complètes (tous les patches)
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

Contient toutes les métadonnées pour chaque patch. **Note :** Ce fichier ne contient pas d'information de split (train/test) car tous les volumes sont préprocessés ensemble. Les splits sont définis dans `train_test_split.json` et appliqués lors du chargement des données.

### Structure

```json
[
  {
    "filename": "stack_000001_patch_00_00.nii.gz",
    "stack_id": "stack_000001",
    "label": 0,
    "position_i": 0,
    "position_j": 0
  },
  ...
]
```

### Champs

- **`filename`** : Nom du fichier `.nii.gz`
- **`stack_id`** : ID du volume d'origine
- **`label`** : `0` (SAIN) ou `1` (MALADE)
- **`position_i`** : Ligne dans la grille (0-indexed)
- **`position_j`** : Colonne dans la grille (0-indexed)

## Fichier `metadata.json`

Contient la configuration complète, les statistiques et les informations de traitement du preprocessing.

### Structure

```json
{
  "version": "v0",
  "created": "2026-01-21T15:05:34.963603",
  "dataset_source": "/path/to/dataset.json",
  "config": {
    "target_height": 256,
    "target_width": 256,
    "target_depth": 32,
    "n_patches_h": 4,
    "n_patches_w": 4,
    "n_patches_per_volume": 16,
    "slice_selection": {
      "method": "intensity"
    },
    "normalization": {
      "method": "z-score",
      "normalize_globally": true,
      ...
    }
  },
  "processing": {
    "n_workers": 32,
    "total_time_seconds": 1234.56,
    "total_time_minutes": 20.58,
    "avg_time_per_volume_seconds": 1.6
  },
  "stats": {
    "total_volumes": 771,
    "total_patches": 12336,
    "errors": 0,
    "value_range": {
      "min": -1000.0,
      "max": 1000.0,
      "mean": 0.0,
      "std": 1.0
    }
  },
  "notes": "Description du preprocessing..."
}
```

### Champs principaux

- **`version`** : Version du preprocessing (ex: "v0", "v1")
- **`created`** : Date/heure de création (ISO format)
- **`dataset_source`** : Chemin vers le dataset JSON source
- **`config`** : Configuration complète (dimensions, méthodes, paramètres)
- **`processing`** : Informations de traitement (workers, temps)
- **`stats`** : Statistiques (volumes, patches, erreurs, plages de valeurs)
- **`notes`** : Notes descriptives du preprocessing

## Format des données dans les fichiers .nii.gz

### Dimensions

Chaque fichier `.nii.gz` contient un patch de dimensions :
- **Spatial** : `H × W` (ex: 256×256)
- **Profondeur** : `D` (ex: 32 slices)
- **Canaux** : `3` (RGB)

Format NIfTI interne : `(D, H, W, 3)` = `(Z, Y, X, C)`

### Type de données

- **Type** : `float32`
- **Normalisation** : Dépend de la config (z-score, min-max, robust, percentile)
- **Mode** : Local (par patch) ou Global (même stats pour tous les patches)
- **Clipping** : Optionnel (valeurs fixes ou redimensionnement de plages)

## Versioning

Chaque preprocessing est identifié par une version (ex: `v0`, `v1`, `v2`). Cela permet de :
- Gérer plusieurs versions de preprocessing avec différents paramètres
- Traçabilité complète des expérimentations
- Comparaison entre différentes approches

Le nom du dossier inclut la version : `preprocessed_{H}x{W}x{D}_{version}`

## Utilisation avec PyTorch

Les patches sont chargés via `train_test_split.json` qui définit quels `stack_id` appartiennent au train/test. Le DataLoader filtre automatiquement les patches selon le split demandé.
