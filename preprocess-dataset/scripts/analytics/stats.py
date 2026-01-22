"""Statistical computations for voxel intensity analysis."""

import numpy as np
from typing import List, Dict


def compute_percentiles_from_histogram(
    bin_centers: np.ndarray,
    total_counts: np.ndarray,
    total_voxels: int,
    percentiles: List[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """Compute percentiles directly from histogram without reconstructing all voxels.
    
    Args:
        bin_centers: Array of bin center values
        total_counts: Array of counts per bin
        total_voxels: Total number of voxels
        percentiles: List of percentile values to compute
        
    Returns:
        Dictionary mapping percentile names to values
    """
    # Compute cumulative distribution
    cumulative = np.cumsum(total_counts)
    cumulative_pct = cumulative / total_voxels
    
    # Find percentiles
    percentile_dict = {}
    for p in percentiles:
        target = p / 100.0
        # Find first bin where cumulative >= target
        idx = np.searchsorted(cumulative_pct, target)
        if idx >= len(bin_centers):
            idx = len(bin_centers) - 1
        percentile_dict[f'p{p}'] = float(bin_centers[idx])
    
    return percentile_dict
