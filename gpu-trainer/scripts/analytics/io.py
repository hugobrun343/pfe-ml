"""I/O utilities for loading training results and preprocessing metadata."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_results(json_path: Path) -> Dict:
    """
    Load training results JSON file.
    
    Args:
        json_path: Path to training_results.json file
        
    Returns:
        Dictionary containing training results
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_preprocessing_metadata(run_dir: Path) -> Optional[Dict]:
    """
    Load preprocessing metadata.json from run directory.
    
    Looks for metadata.json in run_dir/data/preprocessing_metadata.json
    
    Args:
        run_dir: Path to run directory (e.g., _runs/resnet3d-full-dataset_v0_20240121_120000)
        
    Returns:
        Preprocessing metadata dictionary or None if not found
    """
    metadata_path = run_dir / 'data' / 'preprocessing_metadata.json'
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_grid_position(
    position_h: int,
    position_w: int,
    target_height: int,
    target_width: int
) -> Tuple[int, int]:
    """
    Calculate grid position (i, j) from absolute position (position_h, position_w).
    
    Only works for mode 'max' where patches are in a regular grid.
    
    Args:
        position_h: Absolute vertical position (center of patch)
        position_w: Absolute horizontal position (center of patch)
        target_height: Patch height
        target_width: Patch width
        
    Returns:
        Tuple of (i, j) grid position
    """
    # Calculate grid indices from patch centers
    # Patch center at (position_h, position_w) means patch starts at (position_h - target_h//2, position_w - target_w//2)
    # Grid position i = (patch_start_h) // target_height
    i = (position_h - target_height // 2) // target_height
    j = (position_w - target_width // 2) // target_width
    
    return (i, j)
