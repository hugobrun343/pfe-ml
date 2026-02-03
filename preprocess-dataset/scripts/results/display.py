"""Print config and run summaries."""

from pathlib import Path
from typing import Any, Dict


def print_config_summary(cfg: Dict[str, Any], norm_config: Dict[str, Any], version: str) -> None:
    """Print a short preprocessing config summary."""
    h, w, d = cfg.get("target_height"), cfg.get("target_width"), cfg.get("target_depth")
    pe = cfg.get("patch_extraction") or {}
    mode = pe.get("mode", "max")
    ss = cfg.get("slice_selection") or {}
    if isinstance(ss, dict):
        slice_m = ss.get("method", "intensity")
    else:
        slice_m = str(ss)
    norm_m = norm_config.get("method", "z-score")
    print(f"Config {version}: {h}x{w}x{d} | patch={mode} | slice={slice_m} | norm={norm_m}")
    if mode == "top_n":
        print(f"  top_n: n_patches={pe.get('n_patches')}, pool_stride={pe.get('pool_stride')}")


def print_run_summary(
    volume_count: int,
    valid_count: int,
    errors: int,
    elapsed: float,
    output_base: Path,
    n_ppv: int,
) -> None:
    """Print run summary (volumes, errors, time, output)."""
    total_patches = volume_count * n_ppv if volume_count and n_ppv else 0
    print(f"\nDone: {volume_count}/{valid_count} volumes, {total_patches} patches, {errors} errors")
    print(f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Output: {output_base}")
