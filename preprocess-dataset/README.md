# Preprocessing Scripts Structure

## Organisation

Config under `scripts/config/`, entrypoint and pipeline under `scripts/`:

```
scripts/
├── config/                      # YAML configs (v1.1 … v1.6)
├── preprocess_volumes_nii.py    # Preprocessing → patches (--check = run validation at the end)
│
└── preprocessing/               # Preprocessing by feature
    ├── config/   load_context, get_output_dirs
    ├── io/       load_volume, save_patch_nii (NIfTI)
    ├── normalize/ normalize_patch
    ├── patches/  slice_selection, score_map, extraction, positioning, utils
    ├── results/  metadata, write_run_results, display
    ├── run/      load_valid_stacks, process_single_volume, process_all_volumes
    └── check/    run_validation, run_post_check (count, empty, norm, dim)
```

**Voxel intensity analysis** lives in a separate project: see the sibling folder **voxel-analytics**.

## Usage

### Preprocessing

```bash
python scripts/preprocess_volumes_nii.py --config scripts/config/preprocess_config_v1.1.yaml --workers 16
```

**Patch extraction configuration**

Two extraction modes:

1. **Mode `max`**: Extract as many non-overlapping patches as fit in a regular 3D grid.
   - Number of patches is derived from volume and patch sizes.
   - Example: volume 1042×1042×D, patches 256×256×32 → 4×4 grid in H,W and D/32 in depth.

2. **Mode `top_n`**: Extract the N best patches by a 3D score (max-pool over the volume).
   - Picks the most interesting regions (intensity, variance, entropy, gradient).
   - Fixed number of patches, minimum spacing to avoid overlap.
   - Example: `n_patches: 16, scoring_method: "intensity"` → 16 patches at highest-intensity 3D positions.

**Important:** Patches use exact size `target_h × target_w × target_d`. If the volume is not a multiple of patch size, edge pixels are unused. The pipeline errors if the requested number of patches exceeds the theoretical maximum.

### Voxel intensity analysis

**Separate project**: sibling folder **voxel-analytics** (not part of preprocess-dataset).

```bash
cd ../voxel-analytics
python run.py --dataset-json /path/to/dataset.json --data-root /path/to/data --output-dir ./output
python run.py --from-json ./output/voxel_intensity_analysis.json --output-dir ./output
```

See `voxel-analytics/README.md` for options and outputs.

## Modules

### Preprocessing core

- **`io/nii.py`**: NIfTI I/O
  - `load_volume()`: load a .nii.gz volume
  - `save_patch_nii()`: save a patch as .nii.gz

- **`patches/extraction.py`**: 3D patch extraction
  - `extract_patches_max()`: regular grid, no overlap
  - `extract_patches_top_n()`: N best patches from 3D score grid

- **`patches/positioning.py`**: 3D position selection
  - `find_best_patch_positions_3d()`: best 3D centres from score grid (used by `top_n`)

- **`patches/score_map.py`**: 3D score volume for `top_n`
  - `compute_score_volume_3d()`: max-pool 3D over (H,W,D,C) → grid for position selection

- **`patches/utils.py`**: patch helpers
  - `resize_patch()`: resize spatial dimensions

- **`patches/slice_selection.py`**: slice block choice (for `max` mode)
  - `select_best_slices()`: pick best contiguous block of D slices (intensity, variance, entropy, gradient, intensity_range)

- **`normalize/normalize.py`**: normalization
  - `normalize_patch()`: normalize a patch (z-score, min-max, etc.)

### Preprocessing helpers

- **`config/loader.py`**: config loading and resolution
  - `load_context()`: load and validate YAML, resolve paths
  - `get_output_dirs()`: output_base, patches_output
  - `get_slice_selection_method()`, `get_patch_extraction_config()`: slice and patch settings

- **`run/stacks.py`**: dataset loading
  - `load_valid_stacks()`: load JSON, filter valid stacks (SAIN/MALADE, existing nii_path)

- **`run/volume.py`**: single-volume pipeline
  - `process_single_volume()`: load → normalize → extract patches → save

- **`run/pipeline.py`**: batch
  - `process_all_volumes()`: parallel run over all volumes

- **`results/write.py`**, **`results/display.py`**: write metadata, print run summary

- **`check/validate.py`**: patch validation
  - `run_validation()`: count, empty files, norm/dim checks
  - `run_post_check()`: run validation, print, exit(1) on failure (used by `--check`)

See **PIPELINE.md** for the full 3D pipeline and execution order.
