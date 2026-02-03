"""Metadata for preprocessing runs."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config.loader import get_slice_selection_method


def build_metadata(
    config: Dict,
    version: str,
    dataset_path: Path,
    cfg: Dict,
    norm_config: Dict,
    n_workers: int,
    elapsed_time: float,
    volume_count: int,
    n_patches: int,
    errors: int,
    norm_stats: Optional[Dict] = None,
    patch_mode: Optional[str] = None,
    n_patches_config: Optional[int] = None,
    pool_stride: Optional[int] = None,
) -> Dict:
    """Build metadata dictionary for the preprocessing run."""
    slice_method, min_intensity, max_intensity = get_slice_selection_method(cfg)
    patch_extraction_metadata = {}
    if patch_mode:
        patch_extraction_metadata["mode"] = patch_mode
        if patch_mode == "top_n" and n_patches_config is not None:
            patch_extraction_metadata["n_patches"] = n_patches_config
            if pool_stride:
                patch_extraction_metadata["pool_stride"] = pool_stride
    slice_selection_metadata = {"method": slice_method}
    if slice_method == "intensity_range":
        if min_intensity is not None:
            slice_selection_metadata["min_intensity"] = min_intensity
        if max_intensity is not None:
            slice_selection_metadata["max_intensity"] = max_intensity
    metadata = {
        "version": version,
        "created": datetime.now().isoformat(),
        "dataset_source": str(dataset_path),
        "config": {
            "target_height": cfg["target_height"],
            "target_width": cfg["target_width"],
            "target_depth": cfg["target_depth"],
            "patch_extraction": patch_extraction_metadata,
            "n_patches_per_volume": n_patches,
            "slice_selection": slice_selection_metadata,
            "normalization": norm_config,
        },
        "processing": {
            "n_workers": n_workers,
            "total_time_seconds": elapsed_time,
            "total_time_minutes": elapsed_time / 60,
            "avg_time_per_volume_seconds": elapsed_time / volume_count if volume_count > 0 else 0,
        },
        "stats": {
            "total_volumes": volume_count,
            "total_patches": volume_count * n_patches,
            "errors": errors,
        },
        "notes": config.get("notes", ""),
    }
    if norm_stats:
        metadata["stats"]["value_range"] = {k: norm_stats.get(k) for k in ("min", "max", "mean", "std", "median")}
        if "percentile_1" in norm_stats:
            metadata["stats"]["value_range"]["percentile_1"] = norm_stats["percentile_1"]
            metadata["stats"]["value_range"]["percentile_99"] = norm_stats["percentile_99"]
    return metadata
