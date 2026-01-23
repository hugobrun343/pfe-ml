# Format de Sortie : Patches .nii.gz

## Structure des fichiers

Après le preprocessing, chaque patch est sauvegardé comme un fichier `.nii.gz` indépendant.

### Organisation des dossiers

```
output/preprocessed_{H}x{W}x{D}_{version}/
├── patches/
│   ├── stack_000001_patch_000.nii.gz
│   ├── stack_000001_patch_001.nii.gz
│   ├── ...
│   ├── stack_000001_patch_015.nii.gz
│   ├── stack_000002_patch_000.nii.gz
│   └── ...
├── patches_info.json    # Métadonnées complètes (tous les patches)
└── metadata.json         # Configuration et statistiques
```

### Nommage des fichiers

Format : `{stack_id}_patch_{patch_index:03d}.nii.gz`

- `stack_id` : ID du volume d'origine (ex: `stack_000001`)
- `patch_index` : Index séquentiel du patch (0, 1, 2, ...)

**Exemples :**
- `stack_000001_patch_000.nii.gz` → Premier patch du volume
- `stack_000001_patch_015.nii.gz` → 16ème patch du volume (si 16 patches par volume)
- `stack_000002_patch_000.nii.gz` → Premier patch du volume suivant

**Note :** L'ordre des patches dépend du mode d'extraction :
- Mode `max` : Patches extraits en grille régulière (de gauche à droite, de haut en bas)
- Mode `top_n` : Patches triés par score décroissant (du plus intéressant au moins intéressant)

## Fichier `patches_info.json`

Contient toutes les métadonnées pour chaque patch. **Note :** Ce fichier ne contient pas d'information de split (train/test) car tous les volumes sont préprocessés ensemble. Les splits sont définis dans `train_test_split.json` et appliqués lors du chargement des données.

### Structure

```json
[
  {
    "filename": "stack_000001_patch_000.nii.gz",
    "stack_id": "stack_000001",
    "label": 0,
    "position_h": 128,
    "position_w": 128,
    "patch_index": 0
  },
  ...
]
```

### Champs

- **`filename`** : Nom du fichier `.nii.gz`
- **`stack_id`** : ID du volume d'origine
- **`label`** : `0` (SAIN) ou `1` (MALADE)
- **`position_h`** : Position verticale du centre du patch dans le volume original (en pixels)
- **`position_w`** : Position horizontale du centre du patch dans le volume original (en pixels)
- **`patch_index`** : Index séquentiel du patch (0, 1, 2, ...)

**Note :** Les positions `position_h` et `position_w` indiquent le centre du patch dans le volume original. Cela permet de savoir exactement d'où vient chaque patch, même en mode `top_n` où les patches ne sont pas en grille régulière.

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
    "patch_extraction": {
      "mode": "max"
    },
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
  - **`patch_extraction`** : Configuration de l'extraction de patches
    - **`mode`** : `"max"` ou `"top_n"`
    - **`n_patches`** : (si mode `top_n`) Nombre de patches à extraire
    - **`scoring_method`** : (si mode `top_n`) Méthode de scoring (`"intensity"`, `"variance"`, `"entropy"`, `"gradient"`)
- **`processing`** : Informations de traitement (workers, temps)
- **`stats`** : Statistiques (volumes, patches, erreurs, plages de valeurs)
- **`notes`** : Notes descriptives du preprocessing

**Exemple de config avec mode `top_n` :**
```json
"patch_extraction": {
  "mode": "top_n",
  "n_patches": 16,
  "scoring_method": "intensity"
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
