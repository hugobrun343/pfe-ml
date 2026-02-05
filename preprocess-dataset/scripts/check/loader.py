"""Load a single patch from .nii.gz or .npy for validation."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_patch_as_hwdc(path: Path) -> np.ndarray:
    """Load a patch (.nii.gz or .npy) and return it as (H, W, D, C)."""
    path = Path(path)
    suf = path.suffix
    if path.name.endswith(".nii.gz"):
        suf = ".nii.gz"
    if suf == ".npy":
        vol = np.load(str(path)).astype(np.float32)
        return vol
    if suf == ".nii.gz" or str(path).endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel required: pip install nibabel")
        img = nib.load(str(path))
        vol = img.get_fdata()
        if vol.ndim == 4:
            vol = vol.transpose(1, 2, 0, 3)
        elif vol.ndim == 3:
            vol = vol.transpose(1, 2, 0)
            vol = np.expand_dims(vol, axis=-1)
        return vol.astype(np.float32)
    raise ValueError(f"Unsupported patch format: {path}")
