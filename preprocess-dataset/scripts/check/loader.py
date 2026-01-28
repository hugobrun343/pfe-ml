"""Load a single patch from .nii.gz for validation."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_patch_as_hwdc(path: Path) -> np.ndarray:
    """Load a .nii.gz patch and return it as (H, W, D, C)."""
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
