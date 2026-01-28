"""Patch position selection from score volumes (3D)."""

import numpy as np
from typing import List, Tuple


def find_best_patch_positions_3d(
    score_3d: np.ndarray,
    n_patches: int,
    target_h: int,
    target_w: int,
    target_d: int,
    pool_stride: int,
) -> List[Tuple[int, int, int]]:
    """Find the n_patches best 3D positions (h,w,d) from the score_3d grid (H',W',D').
    score_3d comes from max_pool3d. Returns centers in original coords (× pool_stride)."""
    flat = score_3d.ravel()
    order = np.argsort(flat)[::-1]
    Hp, Wp, Dp = score_3d.shape
    min_dist = max(target_h, target_w, target_d)

    def idx_to_xyz(idx):
        i = idx // (Wp * Dp)
        j = (idx % (Wp * Dp)) // Dp
        k = idx % Dp
        h = i * pool_stride + pool_stride // 2
        w = j * pool_stride + pool_stride // 2
        d = k * pool_stride + pool_stride // 2
        return (h, w, d)

    selected = []
    for idx in order:
        if len(selected) >= n_patches:
            break
        h, w, d = idx_to_xyz(idx)
        if selected:
            arr = np.array(selected)
            dist = np.sqrt((h - arr[:, 0]) ** 2 + (w - arr[:, 1]) ** 2 + (d - arr[:, 2]) ** 2)
            if np.any(dist < min_dist):
                continue
        selected.append((h, w, d))

    if len(selected) < n_patches:
        raise ValueError(
            f"Could only find {len(selected)} non-overlapping 3D positions for {n_patches} patches."
        )
    return selected
