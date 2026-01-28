"""Patch extraction (max and top_n)."""

import numpy as np
from typing import List, Tuple, Literal, Optional, Union

from patches.utils import resize_patch
from patches.score_map import compute_score_volume_3d
from patches.positioning import find_best_patch_positions_3d
from patches.slice_selection import select_best_slices


def extract_patches_max(
    vol: np.ndarray,
    target_h: int,
    target_w: int,
    target_d: int,
    slice_method: Literal["intensity", "variance", "entropy", "gradient", "intensity_range"] = "intensity",
    min_intensity: Optional[Union[float, list]] = None,
    max_intensity: Optional[Union[float, list]] = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Extract max patches, selecting best slices within each patch."""
    H, W, D, C = vol.shape
    patches, positions = [], []
    for i in range(H // target_h):
        h_start, h_end = i * target_h, (i + 1) * target_h
        for j in range(W // target_w):
            w_start, w_end = j * target_w, (j + 1) * target_w
            patch_2d = vol[h_start:h_end, w_start:w_end, :, :].copy()
            if patch_2d.shape[0] != target_h or patch_2d.shape[1] != target_w:
                patch_2d = resize_patch(patch_2d, target_h, target_w)
            patch = select_best_slices(patch_2d, target_d, slice_method, min_intensity, max_intensity)
            patches.append(patch)
            positions.append((h_start + target_h // 2, w_start + target_w // 2))
    return patches, positions


def _crop_3d_centered(vol: np.ndarray, h_center: int, w_center: int, d_center: int, th: int, tw: int, td: int) -> np.ndarray:
    """Extract a th×tw×td block centered at (h_center,w_center,d_center), with padding on overflow."""
    H, W, D, C = vol.shape
    patch = np.zeros((th, tw, td, C), dtype=vol.dtype)
    h_lo_src = max(0, h_center - th // 2)
    h_hi_src = min(H, h_center - th // 2 + th)
    w_lo_src = max(0, w_center - tw // 2)
    w_hi_src = min(W, w_center - tw // 2 + tw)
    d_lo_src = max(0, d_center - td // 2)
    d_hi_src = min(D, d_center - td // 2 + td)
    h_lo_dst = h_lo_src - (h_center - th // 2)
    h_hi_dst = h_lo_dst + (h_hi_src - h_lo_src)
    w_lo_dst = w_lo_src - (w_center - tw // 2)
    w_hi_dst = w_lo_dst + (w_hi_src - w_lo_src)
    d_lo_dst = d_lo_src - (d_center - td // 2)
    d_hi_dst = d_lo_dst + (d_hi_src - d_lo_src)
    patch[h_lo_dst:h_hi_dst, w_lo_dst:w_hi_dst, d_lo_dst:d_hi_dst, :] = vol[h_lo_src:h_hi_src, w_lo_src:w_hi_src, d_lo_src:d_hi_src, :]
    return patch


def extract_patches_top_n(
    vol: np.ndarray,
    n_patches: int,
    target_h: int,
    target_w: int,
    target_d: int,
    pool_stride: int = 2,
    slice_method: Literal["intensity", "variance", "entropy", "gradient", "intensity_range"] = "intensity",
    min_intensity: Optional[Union[float, list]] = None,
    max_intensity: Optional[Union[float, list]] = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """top_n: max-pool 3D → 3D centers (h,w,d) → centered target_h×target_w×target_d patch, no slice selection."""
    H, W, D, C = vol.shape
    score_3d = compute_score_volume_3d(vol, pool_stride)
    positions_3d = find_best_patch_positions_3d(
        score_3d, n_patches, target_h, target_w, target_d, pool_stride
    )
    patches = [_crop_3d_centered(vol, h, w, d, target_h, target_w, target_d) for h, w, d in positions_3d]
    return patches, positions_3d
