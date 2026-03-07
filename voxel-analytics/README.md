# voxel-analytics

**Optional exploratory step.** Computes voxel intensity statistics across all NIfTI volumes in the dataset. Used to inform normalization strategy choices in `preprocess-dataset`.

## Purpose

Analyzes the intensity distribution of all 3D volumes without any preprocessing. Uses a two-pass histogram approach for memory efficiency (no need to load all volumes simultaneously). Outputs a JSON summary of global and per-channel statistics, plus distribution and CDF plots.

## When to use it

Run this before `preprocess-dataset` when choosing or validating a normalization method (z-score, min-max, percentile-based, etc.). The output JSON can be reloaded to regenerate plots without re-processing all volumes.

**Prerequisites:** enriched dataset JSON (from `refacto-dataset`) + NIfTI data root directory.

## Usage

Run from inside the `voxel-analytics/` directory:

```bash
# Full analysis: scan all volumes and save results
python run.py \
    --dataset-json /path/to/dataset_final.json \
    --data-root /path/to/data/raw \
    --output-dir ./analysis_output

# Replot from an existing analysis JSON (no re-processing)
python run.py \
    --from-json ./analysis_output/voxel_intensity_analysis.json \
    --output-dir ./analysis_output
```

## Outputs

| File | Description |
|---|---|
| `analysis_output/voxel_intensity_analysis.json` | Per-channel and global statistics (min, max, mean, median, p1, p99, histogram bins) |
| `analysis_output/*.png` | Distribution plots and CDF per channel |

**Key findings on this dataset (771 volumes, 3 channels):**
- Intensity range: 0–15688
- Global mean: 372, median: 180
- p1: 7.8 → p99: 3569 (long tail of high-intensity voxels)

These values are used by `preprocess-dataset` to compute per-stack `p1/p99` and `p5/p95` percentiles for patch normalization.

## Structure

```
voxel-analytics/
├── run.py                    # CLI entrypoint
├── requirements.txt          # nibabel, numpy, matplotlib, tqdm
├── voxel_analytics/
│   ├── io.py                 # load_volume(): NIfTI → (H, W, D, C) numpy array
│   ├── dataset_io.py         # Load dataset JSON, resolve volume paths, save/load analysis JSON
│   ├── processing.py         # Two-pass histogram computation, intensity distribution
│   ├── stats.py              # Percentile estimation from histogram bins
│   └── visualization.py      # Distribution and CDF plots (matplotlib)
└── scripts/
    └── visualization.py      # Standalone replot script
```

## Architecture

The analysis uses a two-pass approach to avoid loading all volumes into memory:

1. **Pass 1** — scan all volumes to determine the global intensity range and allocate histogram bins.
2. **Pass 2** — accumulate per-channel histograms across all volumes.

Statistics (mean, median, percentiles) are derived from the accumulated histogram without storing raw voxel values.
