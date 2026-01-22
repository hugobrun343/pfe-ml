"""Display and logging utilities for preprocessing."""

import sys
from pathlib import Path
from typing import Dict

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "helpers"))

# Helper imports
import helpers.config_loader

# Import functions
get_slice_selection_method = helpers.config_loader.get_slice_selection_method


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
    n_patches = cfg['n_patches_h'] * cfg['n_patches_w']
    slice_method = get_slice_selection_method(cfg)
    
    print(f"\nConfig: {h}x{w}x{d}, {cfg['n_patches_h']}x{cfg['n_patches_w']}={n_patches} patches/volume")
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
