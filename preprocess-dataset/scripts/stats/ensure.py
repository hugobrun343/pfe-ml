"""Ensure stats for normalization: compute if missing, load into context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import STATS_FILENAME_GLOBAL, STATS_FILENAME_P1P99, STATS_FILENAME_P5P95, _default_stats_workers
from . import paths as _paths
from . import io as _io
from . import compute as _compute


def ensure_stats_for_normalization(
    ctx: Dict[str, Any],
    valid_stacks: List[Dict[str, Any]],
    config_path: Path,
    max_workers: Optional[int] = None,
) -> None:
    """Ensure ctx has stats for intensity_global / minmax_p1p99 / minmax_p5p95.

    - Resolve data_dir (config or <project>/data).
    - If stats file missing, or (per-stack) any valid stack lacking an entry: compute (parallel), save, load.
    - Set ctx["stats"] (global record or file->record map) and ctx["stats_type"] ("global" | "per_stack").
    - max_workers: parallel workers for stats computation; default from CPU count.
    """
    method = (ctx.get("norm_config") or {}).get("method")
    if method not in ("intensity_global", "minmax_p1p99", "minmax_p5p95"):
        return

    norm_config = ctx["norm_config"]
    paths = ctx["paths"]
    data_dir = _paths.get_data_dir(norm_config, config_path)
    data_root = paths["data_root"]
    nw = max_workers if max_workers is not None else _default_stats_workers()

    if method == "intensity_global":
        p = data_dir / STATS_FILENAME_GLOBAL
        if not p.exists():
            _compute.compute_global_intensity(valid_stacks, data_root, data_dir, max_workers=nw)
        ctx["stats"] = _io.load_global_intensity(p)
        ctx["stats_type"] = "global"
        return

    if method == "minmax_p1p99":
        p = data_dir / STATS_FILENAME_P1P99
        if not p.exists():
            _compute.compute_stack_p1p99(valid_stacks, data_root, data_dir, max_workers=nw)
        m = _io.load_stats_map(p)
        if not _compute._per_stack_coverage_ok(m, valid_stacks):
            _compute.compute_stack_p1p99(valid_stacks, data_root, data_dir, max_workers=nw)
            m = _io.load_stats_map(p)
        ctx["stats"] = m
        ctx["stats_type"] = "per_stack"
        return

    if method == "minmax_p5p95":
        p = data_dir / STATS_FILENAME_P5P95
        if not p.exists():
            _compute.compute_stack_p5p95(valid_stacks, data_root, data_dir, max_workers=nw)
        m = _io.load_stats_map(p)
        if not _compute._per_stack_coverage_ok(m, valid_stacks):
            _compute.compute_stack_p5p95(valid_stacks, data_root, data_dir, max_workers=nw)
            m = _io.load_stats_map(p)
        ctx["stats"] = m
        ctx["stats_type"] = "per_stack"
        return
