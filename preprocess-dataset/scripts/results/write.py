"""Write patches_info.json and metadata.json to output_base."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def write_run_results(
    ctx: Dict[str, Any],
    output_base: Path,
    patches_info: List[Dict[str, Any]],
    volume_count: int,
    n_ppv: int,
    errors: int,
    elapsed: float,
    config_path: Path,
    n_workers: int,
) -> None:
    """Write patches_info.json and metadata.json under output_base."""
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # patches_info.json: list of patch records (same as FORMAT_NII)
    patches_path = output_base / "patches_info.json"
    with open(patches_path, "w", encoding="utf-8") as f:
        json.dump(patches_info, f, indent=2, ensure_ascii=True)

    # metadata.json
    cfg = ctx.get("cfg") or {}
    paths = ctx.get("paths") or {}
    config = {
        "target_height": cfg.get("target_height"),
        "target_width": cfg.get("target_width"),
        "target_depth": cfg.get("target_depth"),
        "patch_extraction": cfg.get("patch_extraction"),
        "n_patches_per_volume": n_ppv,
        "slice_selection": cfg.get("slice_selection"),
        "normalization": ctx.get("norm_config"),
    }
    total_patches = volume_count * n_ppv if volume_count and n_ppv else 0
    metadata = {
        "version": ctx.get("version", "?"),
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_source": str(paths.get("dataset_json", "")),
        "config": config,
        "processing": {
            "n_workers": n_workers,
            "total_time_seconds": round(elapsed, 2),
            "total_time_minutes": round(elapsed / 60, 2),
            "avg_time_per_volume_seconds": round(elapsed / volume_count, 2) if volume_count else 0,
        },
        "stats": {
            "total_volumes": volume_count,
            "total_patches": total_patches,
            "errors": errors,
        },
        "notes": (ctx.get("config") or {}).get("notes"),
    }
    meta_path = output_base / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)
