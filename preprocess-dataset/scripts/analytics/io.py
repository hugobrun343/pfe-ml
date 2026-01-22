"""I/O utilities for voxel intensity analysis.

Output JSON Format (voxel_intensity_analysis.json):
{
    "metadata": {
        "bins": 1000,
        "n_volumes": 771,
        "dataset_json": "/path/to/dataset.json",
        "data_root": "/path/to/data"
    },
    "global_stats": {
        "min": 0.0,
        "max": 15688.0,
        "mean": 372.52,
        "std": 123.45,
        "median": 180.0,
        "total_voxels": 214312387821
    },
    "percentiles": {
        "p1": 10.5,
        "p5": 25.3,
        "p10": 45.2,
        "p25": 120.0,
        "p50": 180.0,
        "p75": 350.0,
        "p90": 650.0,
        "p95": 850.0,
        "p99": 1200.0
    },
    "histogram": {
        "bin_centers": [0.0, 15.688, 31.376, ...],
        "proportions": [0.15, 0.12, 0.10, ...],
        "counts": [32146858, 25717486, 21431238, ...]
    }
}
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_volume_paths(dataset_json: Path, data_root: Path) -> List[Path]:
    """Load volume paths from dataset JSON.
    
    Uses the same logic as preprocessing: extracts filename from nii_path
    and looks for it in data_root.
    
    Args:
        dataset_json: Path to dataset JSON file
        data_root: Root directory where .nii.gz files are located
        
    Returns:
        List of absolute volume paths
    """
    with open(dataset_json, 'r') as f:
        data = json.load(f)
    
    if 'stacks' not in data:
        raise ValueError(f"Dataset JSON must have 'stacks' key. Found keys: {list(data.keys())}")
    
    volume_paths = []
    
    for stack in data['stacks']:
        if 'nii_path' not in stack:
            continue
        
        # Extract filename from nii_path (same as preprocessing)
        nii_filename = Path(stack['nii_path']).name
        # Convert .nii to .nii.gz
        nii_filename = nii_filename.replace('.nii', '.nii.gz')
        
        # Look in data_root (same as preprocessing)
        vol_path = data_root / nii_filename
        
        if vol_path.exists():
            volume_paths.append(vol_path.resolve())
    
    return volume_paths


def save_analysis_data(
    output_path: Path,
    bin_centers: list,
    proportions: list,
    total_counts: list,
    global_stats: Dict[str, float],
    percentiles: Dict[str, float],
    metadata: Dict[str, Any]
) -> None:
    """Save analysis data to JSON for later plotting.
    
    Args:
        output_path: Path to output JSON file
        bin_centers: List of bin center values
        proportions: List of proportions per bin
        total_counts: List of counts per bin
        global_stats: Dictionary with min, max, mean, std, median, total_voxels
        percentiles: Dictionary mapping percentile names to values
        metadata: Additional metadata (bins, dataset_json, etc.)
    """
    data = {
        'metadata': metadata,
        'global_stats': global_stats,
        'percentiles': percentiles,
        'histogram': {
            'bin_centers': [float(x) for x in bin_centers],
            'proportions': [float(x) for x in proportions],
            'counts': [int(x) for x in total_counts]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Analysis data saved to: {output_path}")


def load_analysis_data(json_path: Path) -> Dict[str, Any]:
    """Load analysis data from JSON.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary with all analysis data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data
