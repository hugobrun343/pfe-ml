"""Processing functions for volume preprocessing."""

import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))
sys.path.insert(0, str(scripts_dir / "helpers"))

# Core imports
import core.io as nii_io
import core.utils
import core.slice_selection
import core.normalization

# Helper imports
import helpers.config_loader

# Import functions
load_volume = nii_io.load_volume
extract_patches = core.utils.extract_patches
resize_patch = core.utils.resize_patch
select_best_slices = core.slice_selection.select_best_slices
compute_stats_on_sample = core.normalization.compute_stats_on_sample
get_slice_selection_method = helpers.config_loader.get_slice_selection_method


def filter_valid_stacks(dataset: dict, data_root: Path) -> List[dict]:
    """
    Filter stacks to keep only valid ones (SAIN/MALADE with existing nii_path).
    
    Args:
        dataset: Dataset dictionary with 'stacks' key
        data_root: Root directory for volume files
        
    Returns:
        List of valid stack dictionaries
    """
    valid_stacks = []
    for stack in dataset['stacks']:
        classe = stack.get('infos', {}).get('Classe', '')
        if classe not in ['SAIN', 'MALADE']:
            continue
        if not stack.get('nii_path'):
            continue
        nii_filename = Path(stack['nii_path']).name
        vol_path = data_root / nii_filename.replace('.nii', '.nii.gz')
        if vol_path.exists():
            valid_stacks.append(stack)
    return valid_stacks


def compute_global_normalization_stats(
    valid_stacks: List[dict], 
    data_root: Path, 
    cfg: Dict, 
    norm_config: Dict
) -> Dict:
    """
    Compute global normalization statistics on all patches.
    
    Args:
        valid_stacks: List of valid stack dictionaries
        data_root: Root directory for volume files
        cfg: Preprocessing configuration
        norm_config: Normalization configuration
        
    Returns:
        Dictionary with global statistics
    """
    print("Computing GLOBAL normalization statistics on ALL patches...")
    all_patches = []
    
    for stack in tqdm(valid_stacks, desc="Loading patches for global stats"):
        try:
            nii_filename = Path(stack['nii_path']).name
            vol_path = data_root / nii_filename.replace('.nii', '.nii.gz')
            if vol_path.exists():
                vol = load_volume(vol_path)
                slice_method = get_slice_selection_method(cfg)
                vol = select_best_slices(vol, cfg['target_depth'], slice_method)
                patches = extract_patches(vol, cfg['n_patches_h'], cfg['n_patches_w'])
                for patch in patches:
                    patch = resize_patch(patch, cfg['target_height'], cfg['target_width'])
                    all_patches.append(patch)
        except Exception:
            pass
    
    if all_patches:
        norm_stats = compute_stats_on_sample(all_patches, norm_config['method'], normalize_globally=True)
        print(f"  Global stats: min={norm_stats.get('global_min', 0):.1f}, max={norm_stats.get('global_max', 0):.1f}")
        print(f"                mean={norm_stats.get('global_mean', 0):.1f}, std={norm_stats.get('global_std', 0):.1f}")
        return norm_stats
    return None


def compute_sample_normalization_stats(
    valid_stacks: List[dict], 
    data_root: Path, 
    cfg: Dict, 
    norm_config: Dict
) -> Dict:
    """
    Compute normalization statistics on a sample of patches (for percentile/robust methods).
    
    Args:
        valid_stacks: List of valid stack dictionaries
        data_root: Root directory for volume files
        cfg: Preprocessing configuration
        norm_config: Normalization configuration
        
    Returns:
        Dictionary with sample statistics
    """
    print("Computing normalization statistics on sample...")
    sample_patches = []
    
    for stack in valid_stacks[:min(10, len(valid_stacks))]:
        try:
            nii_filename = Path(stack['nii_path']).name
            vol_path = data_root / nii_filename.replace('.nii', '.nii.gz')
            if vol_path.exists():
                vol = load_volume(vol_path)
                slice_method = get_slice_selection_method(cfg)
                vol = select_best_slices(vol, cfg['target_depth'], slice_method)
                patches = extract_patches(vol, cfg['n_patches_h'], cfg['n_patches_w'])
                for patch in patches:
                    patch = resize_patch(patch, cfg['target_height'], cfg['target_width'])
                    sample_patches.append(patch)
        except Exception:
            pass
    
    if sample_patches:
        norm_stats = compute_stats_on_sample(sample_patches, norm_config['method'], normalize_globally=False)
        print(f"  Sample stats: min={norm_stats.get('min', 0):.1f}, max={norm_stats.get('max', 0):.1f}")
        return norm_stats
    return None
