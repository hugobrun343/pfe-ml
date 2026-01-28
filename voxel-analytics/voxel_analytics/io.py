"""NIfTI loading for analysis (H,W,D,C format)."""

import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel required: pip install nibabel")


def load_volume(path: Path) -> np.ndarray:
    """Load a .nii.gz volume and convert it to (H, W, D, C)."""
    img = nib.load(str(path))
    vol = img.get_fdata()
    if vol.ndim == 4:
        vol = vol.transpose(1, 2, 0, 3)
    elif vol.ndim == 3:
        vol = vol.transpose(1, 2, 0)
        vol = np.expand_dims(vol, axis=-1)
    return vol.astype(np.float32)
