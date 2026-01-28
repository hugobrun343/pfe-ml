"""Two-pass histograms and intensity distribution."""

import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path

from voxel_analytics.io import load_volume


def _process_one(args):
    vol_path, global_min, global_max, bins = args
    try:
        if not Path(vol_path).exists():
            return None
        vol = load_volume(Path(vol_path))
        flat = vol.flatten()
        stats = {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "median": float(np.median(flat)),
            "count": len(flat),
        }
        if global_min is not None and global_max is not None:
            counts, _ = np.histogram(flat, bins=bins, range=(global_min, global_max))
            stats["histogram"] = counts.tolist()
        else:
            stats["histogram"] = None
        return stats
    except Exception as e:
        print(f"  Warning: Failed {vol_path}: {e}")
        return None


def process_volumes_two_pass(volume_paths: list, bins: int = 1000, n_workers: int = None) -> list:
    """Pass 1: global min/max. Pass 2: histograms over that range. Returns list of stats."""
    n_workers = n_workers or min(8, cpu_count())
    print(f"\nPass 1/2: global min/max ({n_workers} workers)...")
    with Pool(n_workers) as pool:
        pass1 = list(tqdm(
            pool.imap(_process_one, [(p, None, None, bins) for p in volume_paths]),
            total=len(volume_paths),
            desc="Pass 1",
        ))
    pass1 = [s for s in pass1 if s is not None]
    if not pass1:
        raise ValueError("No volumes processed.")
    gmin, gmax = min(s["min"] for s in pass1), max(s["max"] for s in pass1)
    print(f"  Global range: [{gmin:.2f}, {gmax:.2f}]")
    print(f"\nPass 2/2: histograms...")
    with Pool(n_workers) as pool:
        pass2 = list(tqdm(
            pool.imap(_process_one, [(p, gmin, gmax, bins) for p in volume_paths]),
            total=len(volume_paths),
            desc="Pass 2",
        ))
    return [s for s in pass2 if s is not None]


def compute_distribution_from_histograms(all_stats: list, bins: int = 1000) -> tuple:
    """Aggregate histograms, compute bin_centers, proportions, global_stats."""
    valid = [s for s in all_stats if s]
    gmin = min(s["min"] for s in valid)
    gmax = max(s["max"] for s in valid)
    total_voxels = sum(s["count"] for s in valid)
    total_counts = np.zeros(bins, dtype=np.int64)
    for s in valid:
        if s.get("histogram"):
            total_counts += np.array(s["histogram"], dtype=np.int64)
    edges = np.linspace(gmin, gmax, bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    proportions = total_counts / total_voxels
    global_stats = {
        "min": float(gmin),
        "max": float(gmax),
        "mean": float(np.mean([s["mean"] for s in valid])),
        "std": float(np.mean([s["std"] for s in valid])),
        "median": float(np.median([s["median"] for s in valid])),
        "total_voxels": int(total_voxels),
    }
    return bin_centers, proportions, total_counts, global_stats
