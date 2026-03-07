# refacto-dataset

**Step 0 of the pipeline.** Builds the enriched dataset JSON from the raw TSV database and the hierarchical folder structure of two-photon microscopy volumes.

## Purpose

The raw data is organized as a hierarchy of folders: `Class → Sample → Measurement`, where measurement folder names encode biomechanical conditions (e.g., `120_2_D` → pressure 120 mmHg, axial stretch 2, dorsal orientation). This module parses that structure, cross-references it with the TSV metadata, and produces a single enriched JSON file used by all downstream modules.

## Output

**`_dataset-json/dataset_final.json`** — one entry per measurement stack with the following fields:

| Field | Description |
|---|---|
| `stack_id` | Unique identifier |
| `nii_path` | Absolute path to the NIfTI volume |
| `classe` | `SAIN` (healthy, label 0) or `MALADE` (diseased, label 1) |
| `age` | Animal age (weeks) |
| `sex` | `M` or `F` |
| `region` | Vascular region |
| `pressure` | Intraluminal pressure (mmHg) — 80 or 120 |
| `axial_stretch` | Axial stretch ratio |
| `orientation` | `D` (dorsal) or `V` (ventral) |
| `genetic` | Genetic background |

## When to use it

Run once at the start of the project, or when the raw database (TSV or folder structure) changes.

**Prerequisites:** raw TSV file + NIfTI data root directory.

## Structure

```
refacto-dataset/
├── scripts/
│   ├── 01_extract_data.py          # TSV + folder structure → expanded TSV (one row per measurement)
│   ├── 02_clean_dataset_paths.py   # Fix/prefix paths in a JSON dataset
│   ├── 03_enrich_dataset_with_tsv.py  # Merge JSON dataset with TSV metadata
│   ├── 04_check_anomalies.py       # Detect anomalous entries (missing files, unexpected values)
│   ├── 05_analyze_infos.py         # Print dataset statistics (class balance, age distribution, etc.)
│   └── utils.py                    # Folder name parsing (pressure, stretch, orientation extraction)
├── backups/           # Original TSV files (not tracked in git)
├── data_intermediate/ # Intermediate processing outputs (not tracked in git)
└── data_final/        # Final enriched JSON (not tracked in git — canonical copy in _dataset-json/)
```

## Usage

All scripts accept `-h` for help. Run them in order:

```bash
# 1. Expand TSV + folder structure into one row per measurement (TSV → expanded TSV)
python scripts/01_extract_data.py \
    --input backups/database_original.tsv \
    --base-path /path/to/data/root \
    --output data_intermediate/database_extracted.tsv

# 2. Clean absolute paths in the base dataset JSON (remove machine-specific prefix)
python scripts/02_clean_dataset_paths.py \
    --input backups/dataset_original.json \
    --output data_intermediate/dataset_cleaned.json \
    --prefix "/storage/simple/users/.../ds_snapshot_2026-01-11/"

# 3. Enrich cleaned JSON with TSV metadata (age, sex, region, genetic, pressure, etc.)
python scripts/03_enrich_dataset_with_tsv.py \
    --json data_intermediate/dataset_cleaned.json \
    --tsv data_intermediate/database_extracted.tsv \
    --output data_final/dataset_enriched_FINAL.json

# 4. Check for anomalies in the folder structure (missing files, unexpected names)
python scripts/04_check_anomalies.py \
    --base-path /path/to/data/root \
    --tsv backups/database_original.tsv

# 5. Inspect dataset statistics (class balance, age distribution, etc.)
python scripts/05_analyze_infos.py --input data_final/dataset_enriched_FINAL.json
```

## Configuration

The scripts use command-line arguments only (no YAML config). All flags have both long (`--input`) and short (`-i`) forms. Key parameters:

| Flag | Script(s) | Description |
|---|---|---|
| `--input` / `-i` | 01, 02, 05 | Input file (TSV for 01, JSON for 02 and 05) |
| `--output` / `-o` | 01, 02, 03, 04 | Output file path |
| `--base-path` / `-b` | 01, 04 | Data root directory (base path to NIfTI folder structure) |
| `--prefix` / `-p` | 02 | Path prefix to strip from all file paths in the JSON |
| `--json` / `-j` | 03 | Input dataset JSON (for enrichment) |
| `--tsv` / `-t` | 03, 04 | TSV file with metadata or folder paths to check |
