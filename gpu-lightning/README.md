# gpu-lightning

Training Lightning propre, piloté par YAML, avec structure de run alignée sur `_runs`.

## Ce qui est configurable
- modèle (`model.name`)
- batch size (`input.batch_size`)
- epochs (`training.epochs`)
- dataset préprocess + split (`data.preprocessed_dir`, `data.train_test_split_json`)
- projet/run W&B (`wandb.project`, `wandb.run_name`)
- early stopping (`training.early_stopping.*`)

## Valeurs en dur
- `in_channels=3`
- train loader: `num_workers=8`, `prefetch_factor=2`, `pin_memory=True`
- val loader: `num_workers=8`, `prefetch_factor=2`, `pin_memory=False`

## Run folder reproduit
Chaque run crée: `_runs/<run_name>_<timestamp>/`
- `data/`:
  - `train_test_split.json`
  - `preprocessing_metadata.json`
  - `patches_info.json`
  - `original_dataset.json` (si fourni/trouvable)
  - `config.yaml`
  - `wandb_info.json`
- `checkpoints/`:
  - `best_model.pth` (uniquement le meilleur, monitor `val_f1_mean`)
- `results/`:
  - `training_summary.json`
- `analytics/`
- `wandb/`

## Métriques F1
- sortie binaire via `sigmoid(logits) >= 0.5`
- `val_f1_pos` et `val_f1_neg` calculés globalement sur toute l'époque (pas moyenne batch)
- `val_f1_mean = (val_f1_pos + val_f1_neg)/2`
- checkpoint + early stopping monitorent `val_f1_mean`

## Lancer
```bash
python /mnt/pve/work/work-hugo/gpu-lightning/step4_train.py \
  --config /mnt/pve/work/work-hugo/gpu-lightning/config/train_config.yaml
```

## Utiliser le meilleur modèle
```bash
python /mnt/pve/work/work-hugo/gpu-lightning/step5_use_model.py \
  --log-dir /mnt/pve/work/work-hugo/_runs
```
