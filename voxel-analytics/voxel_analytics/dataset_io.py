"""Paths (dataset JSON) and save/load of analysis JSON."""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_volume_paths(dataset_json: Path, data_root: Path) -> List[Path]:
    """List .nii.gz volume paths from dataset JSON (stacks, nii_path keys)."""
    with open(dataset_json, "r") as f:
        data = json.load(f)
    if "stacks" not in data:
        raise ValueError(f"Dataset JSON must have 'stacks' key. Found: {list(data.keys())}")
    paths = []
    for stack in data["stacks"]:
        if "nii_path" not in stack:
            continue
        name = Path(stack["nii_path"]).name.replace(".nii", ".nii.gz")
        p = data_root / name
        if p.exists():
            paths.append(p.resolve())
    return paths


def save_analysis_data(
    output_path: Path,
    bin_centers: list,
    proportions: list,
    total_counts: list,
    global_stats: Dict[str, float],
    percentiles: Dict[str, float],
    metadata: Dict[str, Any],
) -> None:
    """Save analysis result (stats, percentiles, histogram) to JSON."""
    data = {
        "metadata": metadata,
        "global_stats": global_stats,
        "percentiles": percentiles,
        "histogram": {
            "bin_centers": [float(x) for x in bin_centers],
            "proportions": [float(x) for x in proportions],
            "counts": [int(x) for x in total_counts],
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Analysis data saved to: {output_path}")


def load_analysis_data(json_path: Path) -> Dict[str, Any]:
    """Charge le JSON d'analyse."""
    with open(json_path, "r") as f:
        return json.load(f)
