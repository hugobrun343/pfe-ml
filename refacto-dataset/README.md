# Dataset Imagerie Vasculaire

## Structure

```
refacto-dataset/
├── backups/           # Originaux (ne pas toucher)
├── data_intermediate/ # Fichiers de traitement
├── data_final/        # UTILISER CE FICHIER
├── scripts/           # Scripts Python
└── README.md
```

## Fichier Principal

**`data_final/dataset_enriched_FINAL.json`**

## Scripts

Tous acceptent `-h` pour l'aide.

```bash
# Analyser les données
python scripts/analyze_infos.py -i data_final/dataset_enriched_FINAL.json

# Extraire depuis TSV
python scripts/extract_data.py -i backups/database_original.tsv -b /path/data -o output.tsv

# Nettoyer les chemins JSON
python scripts/clean_dataset_paths.py -i input.json -o output.json -p "prefix/"

# Enrichir JSON avec TSV
python scripts/enrich_dataset_with_tsv.py -j input.json -t data.tsv -o output.json

# Vérifier anomalies
python scripts/check_anomalies.py -b /path/data -t database.tsv
```