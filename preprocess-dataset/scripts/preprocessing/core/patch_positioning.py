"""Patch position selection from score maps."""

import numpy as np
from scipy.ndimage import maximum_filter
from typing import List, Tuple


def find_best_patch_positions(
    score_map: np.ndarray,
    n_patches: int,
    target_h: int,
    target_w: int
) -> List[Tuple[int, int]]:
    """
    Find the N best patch positions from a score map.
    
    Uses local maxima detection with minimum distance constraint to avoid overlaps.
    Note: Patches are extracted with exact size (target_h x target_w), so if the volume
    dimensions are not multiples of patch size, some pixels at the edges will be unused.
    
    Args:
        score_map: 2D score map with shape (H, W)
        n_patches: Number of patches to extract
        target_h: Target patch height
        target_w: Target patch width
        
    Returns:
        List of (h_center, w_center) positions for patches
        
    Raises:
        ValueError: If n_patches exceeds maximum theoretical number of non-overlapping patches
    """
    H, W = score_map.shape
    
    # Calculate maximum theoretical number of patches without overlap
    # Note: Uses integer division, so remainder pixels are ignored
    max_patches_h = H // target_h
    max_patches_w = W // target_w
    max_patches_theoretical = max_patches_h * max_patches_w
    
    if n_patches > max_patches_theoretical:
        remainder_h = H % target_h
        remainder_w = W % target_w
        raise ValueError(
            f"Cannot place {n_patches} patches without overlap. "
            f"Maximum possible: {max_patches_theoretical} "
            f"(volume: {H}x{W}, patches: {target_h}x{target_w}, "
            f"unused pixels: {remainder_h}H x {remainder_w}W)"
        )
    
    # Minimum distance between patch centers to avoid overlap
    min_distance = max(target_h, target_w)
    
    # Find local maxima using maximum filter
    # Use a neighborhood size slightly smaller than min_distance
    neighborhood_size = int(min_distance * 0.8)
    if neighborhood_size < 3:
        neighborhood_size = 3
    
    local_maxima = maximum_filter(score_map, size=neighborhood_size)
    peaks_mask = (score_map == local_maxima) & (score_map > 0)
    
    # Get all peak positions and scores
    peak_positions = np.argwhere(peaks_mask)
    peak_scores = score_map[peaks_mask]
    
    # Sort by score (descending)
    sorted_indices = np.argsort(peak_scores)[::-1]
    peak_positions = peak_positions[sorted_indices]
    peak_scores = peak_scores[sorted_indices]
    
    # Select positions iteratively, respecting minimum distance
    selected_positions = []
    
    for pos, score in zip(peak_positions, peak_scores):
        if len(selected_positions) >= n_patches:
            break
        
        h_center, w_center = int(pos[0]), int(pos[1])
        
        # Check that patch fits within volume bounds
        h_start = h_center - target_h // 2
        h_end = h_center + target_h // 2
        w_start = w_center - target_w // 2
        w_end = w_center + target_w // 2
        
        # Adjust if too close to edges
        if h_start < 0:
            h_center = target_h // 2
            h_start = 0
            h_end = target_h
        elif h_end > H:
            h_center = H - target_h // 2
            h_start = H - target_h
            h_end = H
        
        if w_start < 0:
            w_center = target_w // 2
            w_start = 0
            w_end = target_w
        elif w_end > W:
            w_center = W - target_w // 2
            w_start = W - target_w
            w_end = W
        
        # Check minimum distance from already selected positions
        too_close = False
        for selected_h, selected_w in selected_positions:
            dist = np.sqrt((h_center - selected_h)**2 + (w_center - selected_w)**2)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected_positions.append((h_center, w_center))
    
    # If we didn't find enough positions, raise an error
    if len(selected_positions) < n_patches:
        raise ValueError(
            f"Could only find {len(selected_positions)} non-overlapping patch positions, "
            f"but {n_patches} were requested. Try reducing n_patches or using mode 'max'."
        )
    
    return selected_positions
