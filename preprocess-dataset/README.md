# preprocess-dataset

**Step 1 of the pipeline.** Converts raw NIfTI volumes into fixed-size 3D patches ready for training.

## Purpose

Takes the raw 3D NIfTI volumes (~1042×1042×50-200 voxels, 3 channels) and extracts normalized 3D patches of a fixed target size (256×256×32 by default). Each patch is saved as a `.npy` file alongside a JSON index (`patches_info.json`) mapping each patch to its source stack and label.

**Active configuration (v1.9):** `top_n=4` patches per volume using intensity-based slice selection and `minmax_p1p99` normalization → **3084 patches from 771 volumes**.

## When to use it

Run after `refacto-dataset` (or `voxel-analytics` for normalization guidance) and before `split-dataset`.

**Prerequisites:** enriched dataset JSON (`_dataset-json/dataset_final.json`) + NIfTI data root directory.

## Usage

```bash
# Standard run
python scripts/preprocess_volumes_nii.py \
    --config config/preprocess_config_v1.9.yaml \
    --workers 16

# Run preprocessing then validate outputs
python scripts/preprocess_volumes_nii.py \
    --config config/preprocess_config_v1.9.yaml \
    --workers 16 \
    --check
```

The `--check` flag runs a full validation pass after preprocessing (counts, empty file check, normalization range, patch dimensions).

## Configuration

Config files live in `config/` at the module root. Use `preprocess_config_v1.9.yaml` (latest and recommended).

Key parameters:

```yaml
preprocessing:
  target_height: 256        # patch spatial height
  target_width: 256         # patch spatial width
  target_depth: 32          # patch depth (number of slices)
  output_format: npy        # npy (faster I/O for training) or nii.gz

  patch_extraction:
    mode: top_n             # top_n (recommended) or max (full grid)
    n_patches: 4            # number of patches to extract per volume (top_n mode)
    pool_stride: 2          # stride for the 3D score map max-pooling

  slice_selection:
    method: intensity       # intensity | variance | entropy | gradient | intensity_range

  normalization:
    method: minmax_p1p99    # z_score | min_max | minmax_p1p99 | minmax_p5p95
```

### Patch extraction modes

**`top_n`** (recommended): extract the N highest-scoring patches from the volume using a 3D score map. Patches are spaced to avoid overlap. Use this when you want a fixed, controllable number of patches per volume.

**`max`**: extract all non-overlapping patches that fit in a regular 3D grid. The number of patches depends on the volume dimensions. A 1042×1042×D volume at 256×256×32 yields a 4×4×(D/32) grid.

### Normalization methods

| Method | Description |
|---|---|
| `z_score` | Per-patch zero mean / unit variance |
| `min_max` | Per-patch [0, 1] scaling using global min/max |
| `minmax_p1p99` | Per-stack [0, 1] scaling using p1/p99 percentiles (clips outliers) |
| `minmax_p5p95` | Per-stack [0, 1] scaling using p5/p95 percentiles |

Per-stack percentile statistics are precomputed and stored in `data/stack_p1p99.json` and `data/stack_p5p95.json`.

## Outputs

```
<output_dir>/
├── patches/
│   ├── stack_000001_patch_0.npy
│   ├── stack_000001_patch_1.npy
│   └── ...
└── patches_info.json    # [{stack_id, label, filename, patch_index, score}, ...]
```

The `patches_info.json` format is the contract consumed by `split-dataset` and `gpu-lightning`.

## Structure

```
preprocess-dataset/
├── config/
│   ├── preprocess_config.yaml          # Base template (v0)
│   ├── preprocess_config_v1.1.yaml     # First versioned config
│   └── ...
│   └── preprocess_config_v1.9.yaml     # Current recommended config
├── data/
│   ├── global_intensity.json           # Per-channel global min/max across all volumes
│   ├── stack_p1p99.json                # Per-stack, per-channel p1/p99 percentiles
│   └── stack_p5p95.json                # Per-stack, per-channel p5/p95 percentiles
├── logs/                               # Execution logs per config version
└── scripts/
    ├── preprocess_volumes_nii.py       # Main entrypoint
    ├── config/
    │   └── loader.py                   # YAML config loading, path resolution, validation
    ├── io/
    │   └── nii.py                      # NIfTI I/O (load_volume, save_patch_nii)
    ├── normalize/
    │   └── normalize.py                # normalize_patch() — all normalization methods
    ├── patches/
    │   ├── extraction.py               # extract_patches_max(), extract_patches_top_n()
    │   ├── positioning.py              # find_best_patch_positions_3d() — score-based centre selection
    │   ├── score_map.py                # compute_score_volume_3d() — 3D max-pool score grid
    │   ├── slice_selection.py          # select_best_slices() — contiguous depth block selection
    │   └── utils.py                    # resize_patch()
    ├── results/
    │   ├── write.py                    # Write patches_info.json and metadata
    │   ├── metadata.py                 # Run metadata helpers
    │   └── display.py                  # Print run summary
    ├── run/
    │   ├── stacks.py                   # load_valid_stacks() — filter dataset JSON to valid entries
    │   ├── volume.py                   # process_single_volume() — full per-volume pipeline
    │   └── pipeline.py                 # process_all_volumes() — parallel batch processing
    ├── check/
    │   ├── validate.py                 # run_validation(), run_post_check() — count, dim, norm checks
    │   ├── sample_checks.py            # Per-patch sanity checks
    │   ├── loader.py                   # Load patches for validation
    │   └── counts.py                   # Expected vs actual patch count checks
    └── stats/
        ├── compute.py                  # Compute global intensity stats and percentiles
        ├── ensure.py                   # Ensure stats files exist before preprocessing
        ├── io.py                       # Load/save stats JSON files
        ├── paths.py                    # Stats file path resolution
        ├── workers.py                  # Parallel stats computation
        └── constants.py               # Default histogram bins and percentile thresholds
```

## 3D pipeline execution order

For each volume:

1. Load NIfTI → `(H, W, D, C)` float32 array
2. Select best depth slice block (`slice_selection`)
3. Compute 3D score map over the cropped volume (`score_map`)
4. Select top-N patch positions (`positioning`)
5. Extract patches at selected positions (`extraction`)
6. Normalize each patch (`normalize`)
7. Save patches as `.npy` files + append to `patches_info.json`
