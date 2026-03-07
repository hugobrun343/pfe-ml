# gpu-tester

**Optional utility.** Profiles VRAM usage for 3D models across a grid of input resolutions, depths, and batch sizes. Use this to determine the maximum viable batch size for a given model before launching a real training run.

## Purpose

Runs synthetic forward+backward passes on all combinations of model architecture, spatial resolution, depth, and batch size. Records which configurations succeed (fit in VRAM) and which fail (OOM), then summarizes the results. No real data is required — all inputs are synthetic tensors.

## When to use it

Run before `gpu-lightning` when working on a new GPU or adding a new model, to determine the correct `batch_size` in training configs.

**Prerequisites:** CUDA-capable GPU + conda environment.

## Usage

Run all commands from inside the `gpu-tester/` directory:

```bash
# Full grid search (resumes automatically if interrupted)
python scripts/run_grid_search.py run

# Force restart from scratch (ignore previous results)
python scripts/run_grid_search.py run --no-resume

# Print grid search status (how many configs done / remaining)
python scripts/run_grid_search.py info

# Quick validation: test all models with a single config
python scripts/test_models.py

# Targeted test: 256×256×32 input for SE-ResNet3D, ViT3D, ConvNeXt3D
python scripts/test_256x256x32.py

# Analyze saved results
python scripts/analyze_results.py results/grid_search_results.json
python scripts/analyze_results.py results/grid_search_results.json \
    --family ResNet3D \
    --output results/analysis_resnet
```

## Configuration

`config/grid_search_config.yaml` controls the full grid:

```yaml
gpu:
  name: L40S
  vram_gb: 48

grid_search:
  spatial_resolutions: [32, 64, 128, 256, 512, 1024]  # H×W
  depth_sizes: [16, 24, 32, 48, 64]                    # D
  batch_sizes: [4, 8, 12, 16, 20, 24, 32, 48, 64]
  models: [ResNet3D-50, ResNet3D-101, ..., ViT3D-Base, ...]  # 24 total

simulation:
  use_synthetic_data: true
  enable_backward_pass: true
  num_test_iterations: 5

optimization:
  use_mixed_precision: true
  mixed_precision_dtype: float16
```

## Structure

```
gpu-tester/
├── config/
│   └── grid_search_config.yaml      # Grid search parameters
├── results/
│   ├── grid_search_results.json     # Full results (6480 tested configs)
│   ├── test_256x256x32.json         # Targeted results for 256×256×32
│   └── analysis/                    # Summary statistics and per-family analysis
└── scripts/
    ├── run_grid_search.py            # Main grid search entrypoint
    ├── test_models.py                # Quick all-model validation
    ├── test_256x256x32.py            # Targeted 256×256×32 test
    ├── vram_tester.py                # Core VRAM profiling logic
    ├── analyze_results.py            # Results analysis and reporting
    ├── reset_gpu.py                  # Force CUDA cache clear
    ├── utils.py                      # Shared helpers
    └── models/
        ├── resnet3d.py
        ├── seresnet3d.py
        ├── vit3d.py
        ├── convnext3d.py
        ├── efficientnet3d.py
        └── densenet3d.py
```

## Results summary

Grid tested on an L40S (48GB VRAM):
- **6480 total configurations** (24 models × 6 resolutions × 5 depths × 9 batch sizes)
- **38.8% success rate** overall
- VRAM usage ranged from 0.16 GB to 43.5 GB across successful configs
- At 256×256×32, ResNet3D-50 fits batch size 32 comfortably; ViT3D-Large requires batch size ≤ 4

These results were used to set `batch_size` values in the `gpu-lightning` and `gpu-single-channel` training configs.
