"""Utility functions for patch extraction and resizing."""

import numpy as np
from skimage.transform import resize
from typing import List


def extract_patches(vol: np.ndarray, n_h: int, n_w: int) -> List[np.ndarray]:
    """
    Extract patches from volume in a regular grid.
    
    Args:
        vol: Volume array with shape (H, W, D, C)
        n_h: Number of patches in height dimension
        n_w: Number of patches in width dimension
        
    Returns:
        List of patch arrays, each with shape (patch_h, patch_w, D, C)
    """
    H, W, D, C = vol.shape
    patch_h, patch_w = H // n_h, W // n_w
    
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            h_start = i * patch_h
            h_end = (i + 1) * patch_h if i < n_h - 1 else H
            w_start = j * patch_w
            w_end = (j + 1) * patch_w if j < n_w - 1 else W
            patches.append(vol[h_start:h_end, w_start:w_end, :, :])
    
    return patches


def resize_patch(patch: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize patch spatial dimensions (H, W) to target size.
    
    Args:
        patch: Patch array with shape (H, W, D, C)
        target_h: Target height
        target_w: Target width
        
    Returns:
        Resized patch with shape (target_h, target_w, D, C)
    """
    H, W, D, C = patch.shape
    resized = np.zeros((target_h, target_w, D, C), dtype=patch.dtype)
    
    for d in range(D):
        for c in range(C):
            resized[:, :, d, c] = resize(
                patch[:, :, d, c],
                (target_h, target_w),
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
    
    return resized
