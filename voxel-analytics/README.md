# voxel-analytics

Analyse de la distribution des intensités de voxels sur des volumes NIfTI.

Projet **séparé** de preprocess-dataset : pas de dépendance au preprocessing. Utilise nibabel pour charger les volumes.

## Usage

```bash
# Depuis la racine du projet
python -m voxel_analytics.run \
  --dataset-json /path/to/dataset.json \
  --data-root /path/to/data/raw \
  --output-dir ./output

# Replotter depuis un JSON existant
python -m voxel_analytics.run --from-json ./output/voxel_intensity_analysis.json --output-dir ./output
```

## Structure

```
voxel-analytics/
├── voxel_analytics/
│   ├── io.py           # load_volume (NIfTI → H,W,D,C)
│   ├── dataset_io.py  # chemins (dataset JSON) + save/load JSON d'analyse
│   ├── processing.py  # two-pass histograms, distribution
│   ├── stats.py       # percentiles depuis histogramme
│   └── visualization.py # plots distribution, CDF
└── run.py             # entrypoint CLI
```

## Dépendances

numpy, nibabel, matplotlib, tqdm
