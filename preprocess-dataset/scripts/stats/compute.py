"""Compute stats (global intensity, per-stack p1/p99, p5/p95). Parallelized."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

try:
    from ioutils.nii import load_volume
except ImportError:
    load_volume = None  # type: ignore[misc, assignment]

from . import workers as _workers
from . import io as _io
from . import paths as _paths
from .constants import (
    STATS_FILENAME_GLOBAL,
    STATS_FILENAME_P1P99,
    STATS_FILENAME_P5P95,
    _default_stats_workers,
)


def compute_global_intensity(
    valid_stacks: List[Dict[str, Any]],
    data_root: Path,
    data_dir: Path,
    max_workers: Optional[int] = None,
) -> Path:
    """Compute min/max per channel over all stacks (parallel). Write data_dir/global_intensity.json."""
    _paths._ensure_data_dir(data_dir)
    out_path = data_dir / STATS_FILENAME_GLOBAL
    if load_volume is None:
        raise RuntimeError("ioutils.nii.load_volume not available")

    paths = [str(_paths._vol_path(s, data_root)) for s in valid_stacks if _paths._vol_path(s, data_root).exists()]
    if not paths:
        raise ValueError("No valid volumes found for global intensity")

    nw = max_workers if max_workers is not None else _default_stats_workers()
    global_mins: List[float] = []
    global_maxs: List[float] = []
    nch: Optional[int] = None

    with ProcessPoolExecutor(max_workers=nw) as ex:
        fut = {ex.submit(_workers._worker_global_one, p): p for p in paths}
        it = tqdm(as_completed(fut), total=len(fut), desc="Stats global", unit="vol")
        for f in it:
            res = f.result()
            if res is None:
                continue
            C, mins, maxs = res
            if nch is None:
                nch = C
                global_mins = [float("inf")] * C
                global_maxs = [float("-inf")] * C
            for c in range(C):
                global_mins[c] = min(global_mins[c], mins[c])
                global_maxs[c] = max(global_maxs[c], maxs[c])

    if nch is None:
        raise ValueError("No valid volumes found for global intensity")
    channels = [{"channel": c, "lo": global_mins[c], "hi": global_maxs[c]} for c in range(nch)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"channels": channels}, f, indent=2)
    return out_path


def _compute_stack_percentiles(
    valid_stacks: List[Dict[str, Any]],
    data_root: Path,
    data_dir: Path,
    out_name: str,
    low: int,
    high: int,
    max_workers: Optional[int] = None,
) -> Path:
    """Compute per-stack percentile stats (parallel). low/high = 1,99 or 5,95."""
    _paths._ensure_data_dir(data_dir)
    out_path = data_dir / out_name
    if load_volume is None:
        raise RuntimeError("ioutils.nii.load_volume not available")

    tasks = []
    for s in valid_stacks:
        p = _paths._vol_path(s, data_root)
        if not p.exists():
            continue
        tasks.append((str(p), _paths._nii_name(s), low, high))
    if not tasks:
        raise ValueError("No valid volumes found for per-stack stats")

    nw = max_workers if max_workers is not None else _default_stats_workers()
    file_to_rec: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        fut = {ex.submit(_workers._worker_perstack_one, t): t for t in tasks}
        it = tqdm(as_completed(fut), total=len(fut), desc=f"Stats {out_name}", unit="vol")
        for f in it:
            res = f.result()
            if res is not None:
                file_to_rec[res["file"]] = res

    order = [t[1] for t in tasks]
    records = [file_to_rec[n] for n in order if n in file_to_rec]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return out_path


def compute_stack_p1p99(
    valid_stacks: List[Dict[str, Any]],
    data_root: Path,
    data_dir: Path,
    max_workers: Optional[int] = None,
) -> Path:
    """Compute p1/p99 per channel per stack. Write data_dir/stack_p1p99.json."""
    return _compute_stack_percentiles(
        valid_stacks, data_root, data_dir, STATS_FILENAME_P1P99, 1, 99, max_workers
    )


def compute_stack_p5p95(
    valid_stacks: List[Dict[str, Any]],
    data_root: Path,
    data_dir: Path,
    max_workers: Optional[int] = None,
) -> Path:
    """Compute p5/p95 per channel per stack. Write data_dir/stack_p5p95.json."""
    return _compute_stack_percentiles(
        valid_stacks, data_root, data_dir, STATS_FILENAME_P5P95, 5, 95, max_workers
    )


def _per_stack_coverage_ok(stats_map: Dict[str, Dict[str, Any]], valid_stacks: List[Dict[str, Any]]) -> bool:
    """True iff every valid stack has an entry in stats_map."""
    for s in valid_stacks:
        if _io.get_stats_record_for_stack(stats_map, s) is None:
            return False
    return True
