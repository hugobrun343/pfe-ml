# Output Format: .nii.gz Patches

## File structure

After preprocessing, each patch is saved as a standalone `.nii.gz` file.

### Directory layout

```
output/preprocessed_{H}x{W}x{D}_{version}/
├── patches/
│   ├── stack_000001_patch_000.nii.gz
│   ├── stack_000001_patch_001.nii.gz
│   ├── ...
│   ├── stack_000001_patch_015.nii.gz
│   ├── stack_000002_patch_000.nii.gz
│   └── ...
├── patches_info.json    # Full metadata (all patches)
└── metadata.json         # Config and stats
```

### File naming

Format: `{stack_id}_patch_{patch_index:03d}.nii.gz`

- `stack_id`: source volume ID (e.g. `stack_000001`)
- `patch_index`: sequential patch index (0, 1, 2, ...)

**Examples:**
- `stack_000001_patch_000.nii.gz` → first patch of the volume
- `stack_000001_patch_015.nii.gz` → 16th patch (if 16 patches per volume)
- `stack_000002_patch_000.nii.gz` → first patch of the next volume

**Note:** Patch order depends on extraction mode:
- Mode `max`: patches from a regular grid (left to right, top to bottom)
- Mode `top_n`: patches sorted by decreasing score (most to least interesting)

## File `patches_info.json`

Contains metadata for every patch. **Note:** This file has no train/test split; all volumes are preprocessed together. Splits are defined in `train_test_split.json` and applied at load time.

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

### Fields

- **`filename`**: `.nii.gz` filename
- **`stack_id`**: source volume ID
- **`label`**: `0` (SAIN) or `1` (MALADE)
- **`position_h`**: vertical center of the patch in the original volume (pixels)
- **`position_w`**: horizontal center of the patch in the original volume (pixels)
- **`patch_index`**: sequential patch index (0, 1, 2, ...)

In `top_n` mode, **`position_d`** may also be present (depth center). Positions are patch centers in the original volume for traceability.

## File `metadata.json`

Holds full config, stats, and preprocessing info.

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
  "notes": "Preprocessing description..."
}
```

### Main fields

- **`version`**: preprocessing version (e.g. "v0", "v1")
- **`created`**: creation date/time (ISO)
- **`dataset_source`**: path to source dataset JSON
- **`config`**: full config (dimensions, methods, parameters)
  - **`patch_extraction`**: patch extraction settings
    - **`mode`**: `"max"` or `"top_n"`
    - **`n_patches`**: (if `top_n`) number of patches
    - **`scoring_method`**: (if `top_n`) `"intensity"`, `"variance"`, `"entropy"`, `"gradient"`
- **`processing`**: runtime info (workers, time)
- **`stats`**: volumes, patches, errors, value ranges
- **`notes`**: free-form description

**Example config for `top_n`:**
```json
"patch_extraction": {
  "mode": "top_n",
  "n_patches": 16,
  "scoring_method": "intensity"
}
```

## Data format inside .nii.gz files

### Dimensions

Each `.nii.gz` contains one patch with:
- **Spatial**: `H × W` (e.g. 256×256)
- **Depth**: `D` slices (e.g. 32)
- **Channels**: `3` (RGB)

Internal NIfTI layout: `(D, H, W, 3)` = `(Z, Y, X, C)`.

### Data type

- **Type**: `float32`
- **Normalization**: depends on config (z-score, min-max, robust, percentile)
- **Mode**: per-patch or global
- **Clipping**: optional (fixed bounds or range remapping)

## Versioning

Each run is identified by a version (e.g. `v0`, `v1`). This allows:
- Managing several preprocessings with different settings
- Full traceability
- Comparing runs

The output directory name includes the version: `preprocessed_{H}x{W}x{D}_{version}`.

## Use with PyTorch

Patches are loaded according to `train_test_split.json`, which defines which `stack_id`s belong to train/test. The DataLoader filters patches by the requested split.
