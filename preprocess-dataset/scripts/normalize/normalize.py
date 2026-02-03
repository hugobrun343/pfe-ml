"""Normalization logic (clip + scale)."""

import numpy as np
from typing import Any, Dict, Optional, Union, Tuple


def _apply_stats_lo_hi(
    data: np.ndarray,
    stats_record: Dict[str, Any],
) -> np.ndarray:
    """Clip to [lo, hi] per channel then (x - lo) / (hi - lo). Input patch or volume (H,W,D,C)."""
    channels = stats_record.get("channels")
    if not channels:
        raise ValueError("stats_record must have 'channels'")
    out = data.astype(np.float32)
    C = out.shape[-1]
    for ch in channels:
        c = ch.get("channel")
        if c is None or c >= C:
            continue
        lo = float(ch["lo"])
        hi = float(ch["hi"])
        if hi <= lo:
            hi = lo + 1e-8
        out[..., c] = np.clip(out[..., c], lo, hi)
        out[..., c] = (out[..., c] - lo) / (hi - lo)
    return out


def normalize_patch(
    patch: np.ndarray,
    method: str = "z-score",
    clip_min: Optional[Union[float, list]] = None,
    clip_max: Optional[Union[float, list]] = None,
    scale_below_range: Optional[Union[Tuple[float, float], list]] = None,
    scale_above_range: Optional[Union[Tuple[float, float], list]] = None,
    scale_middle_range: Optional[Union[Tuple[float, float], list]] = None,
    stats_record: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Normalize patch or volume (H,W,D,C). stats_record required for intensity_global, minmax_p1p99, minmax_p5p95."""
    patch = patch.astype(np.float32)
    H, W, D, C = patch.shape

    # intensity_global, minmax_p1p99, minmax_p5p95: use stats only, no clip/scale config
    if method in ("intensity_global", "minmax_p1p99", "minmax_p5p95"):
        if stats_record is None:
            raise ValueError(f"method={method} requires stats_record")
        return _apply_stats_lo_hi(patch, stats_record)

    def to_channel_array(value, name: str):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return np.full(C, float(value), dtype=np.float32)
        if isinstance(value, list):
            if len(value) != C:
                raise ValueError(f"{name} list length ({len(value)}) must match channels ({C})")
            return np.array(value, dtype=np.float32)
        raise ValueError(f"{name} must be scalar or list of length {C}")

    def to_channel_ranges(value, name: str):
        if value is None:
            return None, None
        if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (int, float)):
            return np.full(C, float(value[0]), dtype=np.float32), np.full(C, float(value[1]), dtype=np.float32)
        if isinstance(value, list) and len(value) == C:
            mins = np.array([v[0] if isinstance(v, (list, tuple)) else v for v in value], dtype=np.float32)
            maxs = np.array([v[1] if isinstance(v, (list, tuple)) else v for v in value], dtype=np.float32)
            return mins, maxs
        raise ValueError(f"{name} must be [min, max] or list of {C} [min, max]")

    clip_min_arr = to_channel_array(clip_min, "clip_min") if clip_min is not None else None
    clip_max_arr = to_channel_array(clip_max, "clip_max") if clip_max is not None else None
    scale_below_min, scale_below_max = to_channel_ranges(scale_below_range, "scale_below_range")
    scale_middle_min, scale_middle_max = to_channel_ranges(scale_middle_range, "scale_middle_range")
    scale_above_min, scale_above_max = to_channel_ranges(scale_above_range, "scale_above_range")

    if clip_min_arr is not None or clip_max_arr is not None:
        clip_min_arr = clip_min_arr.reshape(1, 1, 1, C) if clip_min_arr is not None else None
        clip_max_arr = clip_max_arr.reshape(1, 1, 1, C) if clip_max_arr is not None else None
        result = patch.copy()

        if clip_max_arr is not None:
            mask_above = result > clip_max_arr
            if np.any(mask_above):
                result = np.clip(result, None, clip_max_arr)
                if scale_above_min is not None and scale_above_max is not None:
                    scale_above_min_arr = scale_above_min.reshape(1, 1, 1, C)
                    result = np.where(mask_above, scale_above_min_arr, result)

        if clip_min_arr is not None and clip_max_arr is not None:
            mask_middle = (result >= clip_min_arr) & (result <= clip_max_arr)
            if np.any(mask_middle) and scale_middle_min is not None and scale_middle_max is not None:
                scale_middle_min_arr = scale_middle_min.reshape(1, 1, 1, C)
                scale_middle_max_arr = scale_middle_max.reshape(1, 1, 1, C)
                diff = clip_max_arr - clip_min_arr
                valid = diff > 1e-8
                scale_range = scale_middle_max_arr - scale_middle_min_arr
                scale = np.divide(scale_range, diff, out=np.zeros_like(scale_range), where=valid)
                normalized = np.where(valid, scale_middle_min_arr + (result - clip_min_arr) * scale, scale_middle_min_arr)
                result = np.where(mask_middle, normalized, result)

        if clip_min_arr is not None:
            mask_below = result < clip_min_arr
            if np.any(mask_below):
                result = np.clip(result, clip_min_arr, None)
                if scale_below_min is not None and scale_below_max is not None:
                    scale_below_min_arr = scale_below_min.reshape(1, 1, 1, C)
                    result = np.where(mask_below, scale_below_min_arr, result)

        return result

    if method == "z-score":
        mean, std = np.mean(patch), max(np.std(patch), 1e-8)
        patch = (patch - mean) / std
    elif method == "min-max":
        vmin, vmax = np.min(patch), np.max(patch)
        patch = (patch - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(patch)
    elif method == "robust":
        median = np.median(patch)
        q75, q25 = np.percentile(patch, 75), np.percentile(patch, 25)
        patch = (patch - median) / max(q75 - q25, 1e-8)
    elif method == "percentile":
        p_low, p_high = np.percentile(patch, 1), np.percentile(patch, 99)
        if p_high > p_low:
            patch = (np.clip(patch, p_low, p_high) - p_low) / (p_high - p_low)
        else:
            patch = np.zeros_like(patch)

    return patch
