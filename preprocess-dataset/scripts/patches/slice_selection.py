"""Slice selection for volume preprocessing."""

import numpy as np
from typing import Literal, Optional, Union
from scipy.ndimage import uniform_filter1d


def _compute_rolling_sum(scores: np.ndarray, window_size: int) -> np.ndarray:
    return uniform_filter1d(scores, size=window_size, mode="constant") * window_size


def select_best_slices(
    vol: np.ndarray,
    n_slices: int,
    method: Literal["intensity", "variance", "entropy", "gradient", "intensity_range"] = "intensity",
    min_intensity: Optional[Union[float, list]] = None,
    max_intensity: Optional[Union[float, list]] = None,
) -> np.ndarray:
    """Select best contiguous block of n_slices from volume (H, W, D, C)."""
    H, W, D, C = vol.shape
    if D <= n_slices:
        if D < n_slices:
            need = n_slices - D
            pad_before = need // 2
            pad_after = need - pad_before
            vol = np.pad(vol, ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)), mode="constant")
        return vol
    if method == "intensity":
        return _select_best_slices_intensity_variance(vol, n_slices, "intensity")
    if method == "variance":
        return _select_best_slices_intensity_variance(vol, n_slices, "variance")
    if method == "entropy":
        return _select_best_slices_entropy(vol, n_slices)
    if method == "gradient":
        return _select_best_slices_gradient(vol, n_slices)
    if method == "intensity_range":
        return _select_best_slices_intensity_range(vol, n_slices, min_intensity, max_intensity)
    raise ValueError(f"Unknown slice selection method: {method}")


def _select_best_slices_intensity_variance(vol: np.ndarray, n_slices: int, method: str) -> np.ndarray:
    H, W, D, C = vol.shape
    if method == "intensity":
        slice_scores = np.sum(vol, axis=(0, 1)).sum(axis=1)
    elif method == "variance":
        slice_scores = np.mean(np.var(vol, axis=(0, 1)), axis=1)
    else:
        raise ValueError(f"Method {method} not supported")
    block_scores = _compute_rolling_sum(slice_scores, n_slices)
    n_valid = D - n_slices + 1
    best_start = int(np.argmax(block_scores[:n_valid]))
    return vol[:, :, best_start : best_start + n_slices, :]


def _select_best_slices_entropy(vol: np.ndarray, n_slices: int) -> np.ndarray:
    H, W, D, C = vol.shape
    vol_reshaped = vol.transpose(0, 1, 2, 3).reshape(H * W, D, C)
    data_min, data_max = float(vol.min()), float(vol.max())
    n_bins = 256
    bin_edges = np.linspace(data_min, data_max, n_bins + 1, dtype=np.float32)
    vol_flat = vol_reshaped.reshape(H * W * D * C)
    bin_indices = np.clip(np.digitize(vol_flat, bin_edges) - 1, 0, n_bins - 1).reshape(H * W, D, C)
    slice_entropies = np.zeros(D, dtype=np.float32)
    for d in range(D):
        hist = np.bincount(bin_indices[:, d, :].flatten(), minlength=n_bins).astype(np.float32)
        hist = hist[hist > 0]
        if len(hist) > 0:
            hist = hist / hist.sum()
            slice_entropies[d] = -np.sum(hist * np.log2(hist + 1e-10))
    block_scores = _compute_rolling_sum(slice_entropies, n_slices)
    n_valid = D - n_slices + 1
    best_start = int(np.argmax(block_scores[:n_valid]))
    return vol[:, :, best_start : best_start + n_slices, :]


def _select_best_slices_gradient(vol: np.ndarray, n_slices: int) -> np.ndarray:
    H, W, D, C = vol.shape
    grad_h = np.abs(np.diff(vol, axis=0, prepend=vol[0:1, :, :, :]))
    grad_w = np.abs(np.diff(vol, axis=1, prepend=vol[:, 0:1, :, :]))
    grad_mag = np.sqrt(grad_h**2 + grad_w**2)
    slice_scores = np.sum(grad_mag, axis=(0, 1, 3))
    block_scores = _compute_rolling_sum(slice_scores, n_slices)
    n_valid = D - n_slices + 1
    best_start = int(np.argmax(block_scores[:n_valid]))
    return vol[:, :, best_start : best_start + n_slices, :]


def _select_best_slices_intensity_range(
    vol: np.ndarray,
    n_slices: int,
    min_intensity: Optional[Union[float, list]],
    max_intensity: Optional[Union[float, list]],
) -> np.ndarray:
    if min_intensity is None or max_intensity is None:
        raise ValueError("min_intensity and max_intensity must be provided")
    H, W, D, C = vol.shape
    min_vals = np.array(min_intensity, dtype=np.float32).reshape(1, 1, 1, C) if isinstance(min_intensity, list) else np.full((1, 1, 1, C), float(min_intensity), dtype=np.float32)
    max_vals = np.array(max_intensity, dtype=np.float32).reshape(1, 1, 1, C) if isinstance(max_intensity, list) else np.full((1, 1, 1, C), float(max_intensity), dtype=np.float32)
    mask = (vol >= min_vals) & (vol <= max_vals)
    slice_scores = np.sum(mask, axis=(0, 1, 3))
    block_scores = _compute_rolling_sum(slice_scores, n_slices)
    n_valid = D - n_slices + 1
    best_start = int(np.argmax(block_scores[:n_valid]))
    return vol[:, :, best_start : best_start + n_slices, :]
