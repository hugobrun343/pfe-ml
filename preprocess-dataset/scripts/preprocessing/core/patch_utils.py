"""Utility functions for patch operations."""

import numpy as np
from skimage.transform import resize


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
