"""Display and logging utilities for preprocessing."""

import sys
from pathlib import Path
from typing import Dict

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "preprocessing" / "helpers"))

# Helper imports
import preprocessing.helpers.config_loader

# Import functions
get_slice_selection_method = preprocessing.helpers.config_loader.get_slice_selection_method
get_patch_extraction_config = preprocessing.helpers.config_loader.get_patch_extraction_config


def print_config_summary(cfg: Dict, norm_config: Dict, version: str) -> None:
    """
    Print configuration summary.
    
    Args:
        cfg: Preprocessing configuration
        norm_config: Normalization configuration
        version: Version string
    """
    h = cfg['target_height']
    w = cfg['target_width']
    d = cfg['target_depth']
    
    # Get patch extraction config
    patch_mode, n_patches, scoring_method = get_patch_extraction_config(cfg)
    slice_method, _, _ = get_slice_selection_method(cfg)
    
    # Format patch extraction info
    if patch_mode == 'max':
        patch_info = "max (all possible patches)"
    else:  # top_n
        patch_info = f"top_n ({n_patches} patches, scoring: {scoring_method})"
    
    print(f"\nConfig: {h}x{w}x{d}, patch extraction: {patch_info}")
    print(f"  Version: {version}")
    print(f"  Slice selection: {slice_method}")
    print(f"  Normalization: {norm_config['method']}")
    
    if norm_config.get('normalize_globally', False):
        print(f"    Mode: GLOBAL (same stats for all patches)")
    else:
        print(f"    Mode: LOCAL (per-patch normalization)")
    
    if norm_config.get('clip_min') is not None or norm_config.get('clip_max') is not None:
        clip_info = f"[{norm_config.get('clip_min')}, {norm_config.get('clip_max')}]"
        if norm_config.get('below_clip_value') is not None:
            clip_info += f", <{norm_config.get('clip_min')} → {norm_config.get('below_clip_value')}"
        elif norm_config.get('scale_below_range') is not None:
            clip_info += f", <{norm_config.get('clip_min')} → {norm_config.get('scale_below_range')}"
        if norm_config.get('above_clip_value') is not None:
            clip_info += f", >{norm_config.get('clip_max')} → {norm_config.get('above_clip_value')}"
        elif norm_config.get('scale_above_range') is not None:
            clip_info += f", >{norm_config.get('clip_max')} → {norm_config.get('scale_above_range')}"
        if norm_config.get('scale_middle_range') is not None:
            clip_info += f", middle → {norm_config.get('scale_middle_range')}"
        print(f"    Clipping: {clip_info}")
