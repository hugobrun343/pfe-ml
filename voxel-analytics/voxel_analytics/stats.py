"""Percentiles from histogram (without reconstructing all voxels)."""

import numpy as np
from typing import List, Dict


def compute_percentiles_from_histogram(
    bin_centers: np.ndarray,
    total_counts: np.ndarray,
    total_voxels: int,
    percentiles: List[int] = None,
) -> Dict[str, float]:
    """Compute percentiles via histogram CDF."""
    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    cum = np.cumsum(total_counts) / total_voxels
    out = {}
    for p in percentiles:
        idx = np.searchsorted(cum, p / 100.0)
        idx = min(idx, len(bin_centers) - 1)
        out[f"p{p}"] = float(bin_centers[idx])
    return out
