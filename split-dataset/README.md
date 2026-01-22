# Dataset Split

Outils pour créer des splits train/test stratifiés à partir du dataset enrichi.

## Structure

```
split-dataset/
├── config/
│   └── split_config.yaml       # Configuration des filtres et paramètres de split
├── scripts/
│   └── create_train_test_split.py  # Script de création du split
└── output/
    ├── run_20260115_143022/    # Chaque run crée un dossier avec timestamp
    │   ├── train_test_split.json
    │   └── split_summary.txt
    └── run_20260115_150512/
        ├── train_test_split.json
        └── split_summary.txt
```

## Usage

```bash
cd /home/brunh/scratch_lirmm-ar/PFE_Fiorio/work/split-dataset

python scripts/create_train_test_split.py \
    --config config/split_config.yaml \
    --input ../refacto-dataset/data_final/dataset_enriched_FINAL.json \
    --output output/train_test_split.json
```

## Configuration

Éditer `config/split_config.yaml` pour modifier :
- **Filtres** : age, pressure, stretch, region, classe, genetic, sex, orientation
- **Split** : test_size (0.2 = 20%), random_seed, stratify_by
- **Output** : generate_summary, format

Tous les filtres supportent : valeur unique, liste, ou null (désactivé).