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
└── helpers/                    # Utilitaires et orchestration
    ├── config_loader.py       # Chargement et validation de la config
    ├── processing.py          # Fonctions de traitement (stats, filtrage)
    ├── metadata.py            # Génération des métadonnées
    └── display.py             # Affichage et logging
```

## Utilisation

```bash
python scripts/preprocess_volumes_nii.py --config config/preprocess_config.yaml
```

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
