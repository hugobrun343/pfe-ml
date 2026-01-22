"""Metadata generation for preprocessing runs."""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "helpers"))

# Helper imports
import helpers.config_loader

# Import functions
get_slice_selection_method = helpers.config_loader.get_slice_selection_method


def build_metadata(
    config: Dict, 
    version: str, 
    dataset_path: Path, 
    cfg: Dict, 
    norm_config: Dict, 
    n_workers: int, 
    elapsed_time: float,
    volume_count: int, 
    n_patches: int, 
    errors: int, 
    norm_stats: Optional[Dict] = None
) -> Dict:
    """
    Build metadata dictionary for the preprocessing run.
    
    Args:
        config: Full configuration dictionary
        version: Version string
        dataset_path: Path to dataset JSON
        cfg: Preprocessing configuration
        norm_config: Normalization configuration
        n_workers: Number of workers used
        elapsed_time: Total processing time in seconds
        volume_count: Number of volumes processed
        n_patches: Number of patches per volume
        errors: Number of errors encountered
        norm_stats: Optional normalization statistics
        
    Returns:
        Metadata dictionary
    """
    slice_method = get_slice_selection_method(cfg)
    elapsed_min = elapsed_time / 60
    
    metadata = {
        'version': version,
        'created': datetime.now().isoformat(),
        'dataset_source': str(dataset_path),
        'config': {
            'target_height': cfg['target_height'],
            'target_width': cfg['target_width'],
            'target_depth': cfg['target_depth'],
            'n_patches_h': cfg['n_patches_h'],
            'n_patches_w': cfg['n_patches_w'],
            'n_patches_per_volume': n_patches,
            'slice_selection': {
                'method': slice_method
            },
            'normalization': norm_config
        },
        'processing': {
            'n_workers': n_workers,
            'total_time_seconds': elapsed_time,
            'total_time_minutes': elapsed_min,
            'avg_time_per_volume_seconds': elapsed_time / volume_count if volume_count > 0 else 0
        },
        'stats': {
            'total_volumes': volume_count,
            'total_patches': volume_count * n_patches,
            'errors': errors
        },
        'notes': config.get('notes', '')
    }
    
    # Add normalization stats if computed
    if norm_stats:
        metadata['stats']['value_range'] = {
            'min': norm_stats.get('min'),
            'max': norm_stats.get('max'),
            'mean': norm_stats.get('mean'),
            'std': norm_stats.get('std'),
            'median': norm_stats.get('median')
        }
        if 'percentile_1' in norm_stats:
            metadata['stats']['value_range']['percentile_1'] = norm_stats['percentile_1']
            metadata['stats']['value_range']['percentile_99'] = norm_stats['percentile_99']
    
    return metadata
