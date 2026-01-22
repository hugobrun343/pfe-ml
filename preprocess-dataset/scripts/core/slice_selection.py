"""Modular slice selection functions for volume preprocessing."""

import numpy as np
from typing import Literal


def select_best_slices(
    vol: np.ndarray,
    n_slices: int,
    method: Literal['intensity', 'variance', 'entropy', 'gradient'] = 'intensity'
) -> np.ndarray:
    """
    Select best contiguous block of n_slices from volume.
    
    Args:
        vol: Volume array (H, W, D, C)
        n_slices: Number of slices to select
        method: Selection method ('intensity', 'variance', 'entropy', 'gradient')
        
    Returns:
        Selected volume block (H, W, n_slices, C)
    """
    H, W, D, C = vol.shape
    
    # If volume has fewer slices than requested, pad it
    if D <= n_slices:
        if D < n_slices:
            pad = ((0, 0), (0, 0), (0, n_slices - D), (0, 0))
            vol = np.pad(vol, pad, mode='constant')
        return vol
    
    best_score = -np.inf
    best_start = 0
    
    # Evaluate each possible contiguous block
    for start in range(D - n_slices + 1):
        block = vol[:, :, start:start + n_slices, :]
        score = _compute_block_score(block, method)
        
        if score > best_score:
            best_score = score
            best_start = start
    
    return vol[:, :, best_start:best_start + n_slices, :]


def _compute_block_score(block: np.ndarray, method: str) -> float:
    """
    Compute score for a block of slices.
    
    Args:
        block: Block array (H, W, D, C)
        method: Scoring method
        
    Returns:
        Score value
    """
    if method == 'intensity':
        return float(np.sum(block))
        
    elif method == 'variance':
        return float(np.var(block))
        
    elif method == 'entropy':
        # Compute entropy from histogram
        hist, _ = np.histogram(block.flatten(), bins=256)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        hist = hist / hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-10)))
        
    elif method == 'gradient':
        # Sum of absolute gradients in spatial dimensions
        grad_h = np.abs(np.diff(block, axis=0)).sum()
        grad_w = np.abs(np.diff(block, axis=1)).sum()
        return float(grad_h + grad_w)
        
    else:
        raise ValueError(f"Unknown slice selection method: {method}")
