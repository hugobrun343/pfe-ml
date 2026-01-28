"""Patch utilities (resize, etc.)."""

import numpy as np
from skimage.transform import resize


def resize_patch(patch: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize patch spatial dimensions (H, W) to target size."""
    H, W, D, C = patch.shape
    patch_flat = patch.reshape(H, W, D * C)
    resized_flat = np.zeros((target_h, target_w, D * C), dtype=patch.dtype)
    for i in range(D * C):
        resized_flat[:, :, i] = resize(
            patch_flat[:, :, i],
            (target_h, target_w),
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
    return resized_flat.reshape(target_h, target_w, D, C)
