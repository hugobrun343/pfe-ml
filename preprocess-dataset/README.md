# Preprocessing Scripts Structure

## Organisation

```
scripts/
├── preprocess_volumes_nii.py  # Script principal (point d'entrée)
├── dataset_nii.py              # Dataset PyTorch pour charger les patches
│
├── preprocessing/               # Tout le code de preprocessing
│   ├── core/                   # Fonctions fondamentales
│   │   ├── io.py              # Chargement/sauvegarde NIfTI
│   │   ├── patch_extraction.py # Extraction de patches (max/top_n)
│   │   ├── patch_positioning.py # Sélection des meilleures positions
│   │   ├── patch_utils.py     # Utilitaires (redimensionnement)
│   │   ├── score_map.py        # Calcul des cartes de score
│   │   ├── slice_selection.py # Sélection des meilleures slices
│   │   └── normalization.py   # Normalisation et clipping
│   │
│   └── helpers/                # Utilitaires et orchestration
│       ├── config_loader.py   # Chargement et validation de la config
│       ├── processing.py      # Fonctions de traitement (stats, filtrage)
│       ├── metadata.py        # Génération des métadonnées
│       └── display.py         # Affichage et logging
│
└── analytics/                  # Analyse des données
    ├── analyze_voxel_intensities.py  # Script principal d'analyse
    ├── io.py                  # I/O (chargement paths, sauvegarde/chargement JSON)
    ├── processing.py          # Traitement des volumes (multiprocessing)
    ├── stats.py               # Calcul des percentiles
    └── visualization.py        # Génération des plots
```

## Utilisation

### Preprocessing principal

```bash
python scripts/preprocess_volumes_nii.py --config config/preprocess_config.yaml
```

**Configuration du patch extraction :**

Le preprocessing supporte deux modes d'extraction de patches :

1. **Mode `max`** : Extrait le maximum de patches possibles sans chevauchement en grille régulière
   - Calcule automatiquement le nombre de patches qui rentrent dans le volume
   - Exemple : volume 1042×1042, patches 256×256 → 4×4=16 patches (18 pixels non utilisés de chaque côté)

2. **Mode `top_n`** : Extrait les N meilleurs patches selon une méthode de scoring
   - Sélectionne les zones les plus intéressantes (intensity, variance, entropy, gradient)
   - Contrôle précis du nombre de patches
   - Exemple : `n_patches: 16, scoring_method: "intensity"` → 16 patches aux positions les plus intenses

**Note importante :** Les patches sont extraits avec une taille exacte (target_h × target_w). Si les dimensions du volume ne sont pas des multiples de la taille des patches, certains pixels aux bords seront non utilisés. Le code lève une erreur si le nombre de patches demandé dépasse le maximum théorique possible.

**Configuration du patch extraction :**

Le preprocessing supporte deux modes d'extraction de patches :

1. **Mode `max`** : Extrait le maximum de patches possibles sans chevauchement en grille régulière
   - Calcule automatiquement le nombre de patches qui rentrent dans le volume
   - Exemple : volume 1042×1042, patches 256×256 → 4×4=16 patches (18 pixels non utilisés de chaque côté)

2. **Mode `top_n`** : Extrait les N meilleurs patches selon une méthode de scoring
   - Sélectionne les zones les plus intéressantes (intensity, variance, entropy, gradient)
   - Contrôle précis du nombre de patches
   - Exemple : `n_patches: 16, scoring_method: "intensity"` → 16 patches aux positions les plus intenses

**Note importante :** Les patches sont extraits avec une taille exacte (target_h × target_w). Si les dimensions du volume ne sont pas des multiples de la taille des patches, certains pixels aux bords seront non utilisés. Le code lève une erreur si le nombre de patches demandé dépasse le maximum théorique possible.

### Analyse des intensités de voxels

Avant le preprocessing, tu peux analyser la distribution des intensités de voxels pour décider des paramètres de clipping/normalisation :

**Analyse complète (traitement des volumes) :**
```bash
python scripts/analytics/analyze_voxel_intensities.py \
    --dataset-json /mnt/pve/ds_shared/dataset_enriched_FINAL.json \
    --data-root /mnt/pve/ds_shared/data/raw \
    --output-dir ./analysis_output
```

**Replotter depuis un JSON existant :**
```bash
python scripts/analytics/analyze_voxel_intensities.py \
    --from-json ./analysis_output/voxel_intensity_analysis.json \
    --output-dir ./analysis_output
```

**Options :**
- `--dataset-json` : (requis si pas `--from-json`) JSON avec les chemins des volumes
- `--data-root` : (requis si pas `--from-json`) Répertoire où se trouvent les fichiers .nii.gz
- `--output-dir` : Répertoire de sortie pour les plots et le JSON (défaut: même dossier que le JSON)
- `--bins` : Nombre de bins pour l'histogramme (défaut: 1000)
- `--workers` : Nombre de workers (défaut: min(8, CPU count) pour éviter les problèmes de RAM)
- `--from-json` : Replotter depuis un JSON existant (sans retraiter les volumes)

**Sorties :**
- `voxel_intensity_distribution.png` : Distribution des intensités (échelle linéaire et log)
- `voxel_intensity_statistics.png` : Percentiles et fonction de distribution cumulative (CDF)
- `voxel_intensity_analysis.json` : Toutes les données pour replotter plus tard

## Modules

### Preprocessing Core (Fonctions fondamentales)

- **`io.py`** : Opérations I/O sur fichiers NIfTI
  - `load_volume()` : Charger un volume .nii.gz
  - `save_patch_nii()` : Sauvegarder un patch en .nii.gz

- **`patch_extraction.py`** : Extraction de patches
  - `extract_patches_max()` : Extraire le maximum de patches sans chevauchement (grille régulière)
  - `extract_patches_top_n()` : Extraire les N meilleurs patches selon une méthode de scoring

- **`patch_positioning.py`** : Sélection des positions de patches
  - `find_best_patch_positions()` : Trouver les meilleures positions à partir d'une carte de score

- **`score_map.py`** : Calcul des cartes de score
  - `compute_score_map()` : Calculer une carte 2D de score (intensity, variance, entropy, gradient)

- **`patch_utils.py`** : Utilitaires de patches
  - `resize_patch()` : Redimensionner un patch

- **`slice_selection.py`** : Sélection de slices
  - `select_best_slices()` : Sélectionner le meilleur bloc de slices (intensity, variance, entropy, gradient, intensity_range)

- **`normalization.py`** : Normalisation
  - `normalize_patch()` : Normaliser un patch
  - `get_normalization_config()` : Extraire la config de normalisation
  - `compute_stats_on_sample()` : Calculer les statistiques

### Preprocessing Helpers (Utilitaires)

- **`config_loader.py`** : Gestion de la configuration
  - `load_config()` : Charger et valider la config YAML
  - `resolve_paths()` : Résoudre les chemins (absolus/relatifs)
  - `get_slice_selection_method()` : Extraire la méthode de sélection de slices
  - `get_patch_extraction_config()` : Extraire la config d'extraction de patches (mode max/top_n)

- **`processing.py`** : Fonctions de traitement
  - `filter_valid_stacks()` : Filtrer les stacks valides
  - `compute_global_normalization_stats()` : Stats globales de normalisation
  - `compute_sample_normalization_stats()` : Stats sur échantillon

- **`metadata.py`** : Génération de métadonnées
  - `build_metadata()` : Construire le dictionnaire de métadonnées

- **`display.py`** : Affichage
  - `print_config_summary()` : Afficher le résumé de la config
