# Preprocessing Scripts Structure

## Organisation

```
scripts/
├── preprocess_volumes_nii.py  # Script principal (point d'entrée)
│
├── core/                       # Fonctions fondamentales du preprocessing
│   ├── io.py                  # Chargement/sauvegarde NIfTI
│   ├── utils.py               # Extraction de patches, redimensionnement
│   ├── slice_selection.py     # Sélection des meilleures slices
│   └── normalization.py       # Normalisation et clipping
│
├── helpers/                    # Utilitaires et orchestration
│   ├── config_loader.py       # Chargement et validation de la config
│   ├── processing.py          # Fonctions de traitement (stats, filtrage)
│   ├── metadata.py            # Génération des métadonnées
│   └── display.py             # Affichage et logging
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

### Core (Fonctions fondamentales)

- **`io.py`** : Opérations I/O sur fichiers NIfTI
  - `load_volume()` : Charger un volume .nii.gz
  - `save_patch_nii()` : Sauvegarder un patch en .nii.gz

- **`utils.py`** : Utilitaires de traitement
  - `extract_patches()` : Extraire des patches en grille
  - `resize_patch()` : Redimensionner un patch

- **`slice_selection.py`** : Sélection de slices
  - `select_best_slices()` : Sélectionner le meilleur bloc de slices

- **`normalization.py`** : Normalisation
  - `normalize_patch()` : Normaliser un patch
  - `get_normalization_config()` : Extraire la config de normalisation
  - `compute_stats_on_sample()` : Calculer les statistiques

### Helpers (Utilitaires)

- **`config_loader.py`** : Gestion de la configuration
  - `load_config()` : Charger et valider la config YAML
  - `resolve_paths()` : Résoudre les chemins (absolus/relatifs)
  - `get_slice_selection_method()` : Extraire la méthode de sélection

- **`processing.py`** : Fonctions de traitement
  - `filter_valid_stacks()` : Filtrer les stacks valides
  - `compute_global_normalization_stats()` : Stats globales
  - `compute_sample_normalization_stats()` : Stats sur échantillon

- **`metadata.py`** : Génération de métadonnées
  - `build_metadata()` : Construire le dictionnaire de métadonnées

- **`display.py`** : Affichage
  - `print_config_summary()` : Afficher le résumé de la config
