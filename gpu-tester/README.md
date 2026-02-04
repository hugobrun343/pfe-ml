# gpu-tester

Test VRAM usage for 3D CNN/Transformer models. Run grid search or targeted tests to find max batch size per model for given input dimensions.

## Structure

```
gpu-tester/
├── config/          # YAML configs (grid search)
├── results/         # JSON/CSV outputs
├── scripts/         # Code + entry points (run from project root)
│   ├── models/      # 3D models (ResNet, ViT, ConvNeXt, SE-ResNet, EfficientNet)
│   ├── analyze/     # Result analysis & plots
│   ├── run_grid_search.py
│   ├── test_models.py
│   ├── test_256x256x32.py
│   ├── analyze_results.py
│   ├── grid_search_runner.py
│   ├── vram_tester.py
│   └── utils.py
└── README.md
```

## Usage

Run from project root (`gpu-tester/`):

```bash
# Grid search (full or resumed)
python scripts/run_grid_search.py run
python scripts/run_grid_search.py run --no-resume
python scripts/run_grid_search.py info

# Test all models (quick validation)
python scripts/test_models.py

# VRAM test for 256×256×32×3 (SE-ResNet3D, ViT3D, ConvNeXt3D)
python scripts/test_256x256x32.py

# Analyze results
python scripts/analyze_results.py results/grid_search_results.json
python scripts/analyze_results.py results/grid_search_results.json --family ResNet3D --output results/analysis_resnet
```

## Config

- `config/grid_search_config.yaml` — spatial resolutions, depths, batch sizes, models, simulation params.
