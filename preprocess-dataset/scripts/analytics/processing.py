"""Processing functions for voxel intensity analysis."""

import sys
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))

# Import from core (same as preprocess_volumes_nii.py)
import core.io as nii_io
load_volume = nii_io.load_volume


def process_single_volume(args):
    """Process a single volume and return intensity statistics.
    
    Computes histogram directly to avoid sending all voxels through multiprocessing.
        
    Args:
        args: Tuple of (vol_path, global_min, global_max, bins)
        
    Returns:
        Dictionary with stats and histogram counts
    """
    vol_path, global_min, global_max, bins = args
    
    try:
        if not vol_path.exists():
            return None
        
        # Load volume
        volume = load_volume(vol_path)
        
        # Flatten
        flat = volume.flatten()
        
        # Compute stats
        stats = {
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'median': float(np.median(flat)),
            'count': len(flat)
        }
        
        # Compute histogram 
        if global_min is not None and global_max is not None:
            counts, _ = np.histogram(flat, bins=bins, range=(global_min, global_max))
            stats['histogram'] = counts.tolist()
        else:
            # First pass: just return stats, no histogram yet (global min/max not known yet)
            stats['histogram'] = None
        
        return stats
    except Exception as e:
        print(f"  Warning: Failed to process {vol_path}: {e}")
        return None


def process_volumes_two_pass(volume_paths: list, bins: int = 1000, n_workers: int = None) -> list:
    """Process volumes in two passes: first get global min/max, then compute histograms.
    
    Args:
        volume_paths: List of volume paths to process
        bins: Number of bins for histogram
        n_workers: Number of workers (default: min(8, CPU count))
        
    Returns:
        List of statistics dictionaries
    """
    if n_workers is None:
        n_workers = min(8, cpu_count())
    
    # FIRST PASS: get global min/max
    print(f"\nPass 1/2: Computing global min/max with {n_workers} workers...")
    
    with Pool(n_workers) as pool:
        all_stats_pass1 = list(tqdm(
            pool.imap(process_single_volume, [(vp, None, None, bins) for vp in volume_paths]),
            total=len(volume_paths),
            desc="Pass 1: Computing stats"
        ))
    
    # Filter out None results
    all_stats_pass1 = [s for s in all_stats_pass1 if s is not None]
    
    if len(all_stats_pass1) == 0:
        raise ValueError("No volumes processed successfully!")
    
    # Get global min/max
    global_min = min(s['min'] for s in all_stats_pass1)
    global_max = max(s['max'] for s in all_stats_pass1)
    print(f"  Global range: [{global_min:.2f}, {global_max:.2f}]")
    
    # SECOND PASS: compute histograms with known range
    print(f"\nPass 2/2: Computing histograms with {n_workers} workers...")
    with Pool(n_workers) as pool:
        all_stats = list(tqdm(
            pool.imap(process_single_volume, [(vp, global_min, global_max, bins) for vp in volume_paths]),
            total=len(volume_paths),
            desc="Pass 2: Computing histograms"
        ))
    
    # Filter out None results
    all_stats = [s for s in all_stats if s is not None]
    
    if len(all_stats) == 0:
        raise ValueError("No volumes processed successfully!")
    
    return all_stats


def compute_distribution_from_histograms(all_stats: list, bins: int = 1000) -> tuple:
    """Compute intensity distribution from accumulated histograms.
    
    Args:
        all_stats: List of statistics dictionaries with histograms
        bins: Number of bins for histogram
        
    Returns:
        Tuple of (bin_centers, proportions, total_counts, global_stats_dict)
    """
    print(f"\nComputing distribution from {len(all_stats)} volumes...")
    
    # Compute global stats
    valid_stats = [s for s in all_stats if s]
    global_min = min(s['min'] for s in valid_stats)
    global_max = max(s['max'] for s in valid_stats)
    global_mean = np.mean([s['mean'] for s in valid_stats])
    global_std = np.mean([s['std'] for s in valid_stats])
    global_median = np.median([s['median'] for s in valid_stats])
    total_voxels = sum(s['count'] for s in valid_stats)
    
    print(f"  Global min: {global_min:.2f}")
    print(f"  Global max: {global_max:.2f}")
    print(f"  Global mean: {global_mean:.2f}")
    print(f"  Global std: {global_std:.2f}")
    print(f"  Global median: {global_median:.2f}")
    print(f"  Total voxels: {total_voxels:,}")
    
    # Sum all histograms
    total_counts = np.zeros(bins, dtype=np.int64)
    for stats in valid_stats:
        if stats.get('histogram'):
            total_counts += np.array(stats['histogram'], dtype=np.int64)
    
    # Compute bin centers
    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    proportions = total_counts / total_voxels
    
    global_stats = {
        'min': float(global_min),
        'max': float(global_max),
        'mean': float(global_mean),
        'std': float(global_std),
        'median': float(global_median),
        'total_voxels': int(total_voxels)
    }
    
    return bin_centers, proportions, total_counts, global_stats
