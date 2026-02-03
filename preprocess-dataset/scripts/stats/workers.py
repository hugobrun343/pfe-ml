"""Process-pool workers for stats computation. Top-level functions for pickle."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _worker_global_one(path_str: str) -> Optional[Tuple[int, List[float], List[float]]]:
    """Load one volume, return (nch, mins, maxs) per channel."""
    try:
        from ioutils.nii import load_volume
    except ImportError:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    vol = load_volume(p)
    if vol.ndim != 4:
        return None
    _, _, _, C = vol.shape
    mins = [float("inf")] * C
    maxs = [float("-inf")] * C
    for c in range(C):
        ch = vol[..., c]
        v = np.isfinite(ch)
        if not np.any(v):
            continue
        mn, mx = float(np.min(ch[v])), float(np.max(ch[v]))
        mins[c] = mn
        maxs[c] = mx
    return (C, mins, maxs)


def _worker_perstack_one(args: Tuple[str, str, int, int]) -> Optional[Dict[str, Any]]:
    """Load one volume, compute percentile stats. (path, nii_name, low, high)."""
    path_str, nii_name, low, high = args
    try:
        from ioutils.nii import load_volume
    except ImportError:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    vol = load_volume(p)
    if vol.ndim != 4:
        return None
    _, _, _, C = vol.shape
    ch_list = []
    for c in range(C):
        arr = vol[..., c].ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ch_list.append({"channel": c, "lo": 0.0, "hi": 1.0})
        else:
            lo = float(np.percentile(arr, low))
            hi = float(np.percentile(arr, high))
            if hi <= lo:
                hi = lo + 1e-6
            ch_list.append({"channel": c, "lo": lo, "hi": hi})
    return {"file": nii_name, "channels": ch_list}
