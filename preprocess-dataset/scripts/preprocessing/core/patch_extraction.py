"""Patch extraction functions for volume preprocessing.

Supports two modes:
- 'max': Extract maximum number of patches without overlap in a regular grid
- 'top_n': Extract N best patches based on a scoring method (intensity, variance, entropy, gradient)

Note: Patches are extracted with exact size (target_h x target_w). If volume dimensions
are not multiples of patch size, some pixels at the edges will be unused.
Example: volume 1042x1042, patches 256x256 → 4x4=16 patches, 18 pixels unused on each side.
"""

import numpy as np
from typing import List, Tuple, Literal

from preprocessing.core.patch_utils import resize_patch
from preprocessing.core.score_map import compute_score_map
from preprocessing.core.patch_positioning import find_best_patch_positions


def extract_patches_max(
    vol: np.ndarray,
    target_h: int,
    target_w: int
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract maximum number of patches without overlap in a regular grid.
    
    Note: Uses integer division to determine number of patches. If volume dimensions
    are not multiples of patch size, remainder pixels are ignored.
    Example: volume 1042x1042, patches 256x256 → 4x4=16 patches, 18 pixels unused.
    
    Args:
        vol: Volume array with shape (H, W, D, C)
        target_h: Target patch height
        target_w: Target patch width
        
    Returns:
        Tuple of (list of patches, list of (h_center, w_center) positions)
        Each patch has shape (target_h, target_w, D, C)
    """
    H, W, D, C = vol.shape
    
    # Calculate maximum number of patches that fit (integer division)
    n_h_max = H // target_h
    n_w_max = W // target_w
    
    patches = []
    positions = []
    
    for i in range(n_h_max):
        for j in range(n_w_max):
            h_start = i * target_h
            h_end = h_start + target_h
            w_start = j * target_w
            w_end = w_start + target_w
            
            patch = vol[h_start:h_end, w_start:w_end, :, :]
            
            # Resize if needed (shouldn't be needed, but safety check)
            if patch.shape[0] != target_h or patch.shape[1] != target_w:
                patch = resize_patch(patch, target_h, target_w)
            
            patches.append(patch)
            positions.append((h_start + target_h // 2, w_start + target_w // 2))
    
    return patches, positions


def extract_patches_top_n(
    vol: np.ndarray,
    n_patches: int,
    target_h: int,
    target_w: int,
    scoring_method: Literal['intensity', 'variance', 'entropy', 'gradient']
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract N best patches based on a scoring method.
    
    Args:
        vol: Volume array with shape (H, W, D, C)
        n_patches: Number of patches to extract
        target_h: Target patch height
        target_w: Target patch width
        scoring_method: Method to score regions ('intensity', 'variance', 'entropy', 'gradient')
        
    Returns:
        Tuple of (list of patches, list of (h_center, w_center) positions)
        Each patch has shape (target_h, target_w, D, C)
        
    Raises:
        ValueError: If n_patches exceeds maximum theoretical number of non-overlapping patches
    """
    H, W, D, C = vol.shape
    
    # Compute 2D score map
    score_map = compute_score_map(vol, scoring_method)
    
    # Find best positions
    positions = find_best_patch_positions(score_map, n_patches, target_h, target_w)
    
    # Extract patches at these positions
    patches = []
    for h_center, w_center in positions:
        h_start = h_center - target_h // 2
        h_end = h_start + target_h
        w_start = w_center - target_w // 2
        w_end = w_start + target_w
        
        # Ensure bounds are valid (should be after find_best_patch_positions, but safety check)
        h_start = max(0, h_start)
        h_end = min(H, h_end)
        w_start = max(0, w_start)
        w_end = min(W, w_end)
        
        patch = vol[h_start:h_end, w_start:w_end, :, :]
        
        # Resize if needed (shouldn't be needed, but safety check)
        if patch.shape[0] != target_h or patch.shape[1] != target_w:
            patch = resize_patch(patch, target_h, target_w)
        
        patches.append(patch)
    
    return patches, positions
