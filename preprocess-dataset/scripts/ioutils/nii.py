"""Load/save NIfTI volumes and patches."""

import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")

_DEFAULT_AFFINE = np.eye(4, dtype=np.float32)


def load_volume(path: Path) -> np.ndarray:
    """Load NIfTI (.nii.gz) volume and convert to (H, W, D, C)."""
    img = nib.load(str(path))
    vol = img.get_fdata()
    if vol.ndim == 4:
        vol = vol.transpose(1, 2, 0, 3)
    elif vol.ndim == 3:
        vol = vol.transpose(1, 2, 0)
        vol = np.expand_dims(vol, axis=-1)
    return vol.astype(np.float32)


def save_patch_nii(patch: np.ndarray, output_path: Path, affine: np.ndarray = None) -> None:
    """Save patch as compressed .nii.gz file."""
    if patch.size == 0:
        raise ValueError(f"Cannot save empty patch to {output_path}")
    if not np.isfinite(patch).all():
        raise ValueError(f"Patch contains non-finite values: {output_path}")
    patch_nii = np.ascontiguousarray(np.transpose(patch, (2, 0, 1, 3)), dtype=np.float32)
    if affine is None:
        affine = _DEFAULT_AFFINE
    img = nib.Nifti1Image(patch_nii, affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise IOError(f"Failed to save patch: {output_path}")


def save_patch_npy(patch: np.ndarray, output_path: Path) -> None:
    """Save patch as .npy file (raw numpy array)."""
    if patch.size == 0:
        raise ValueError(f"Cannot save empty patch to {output_path}")
    if not np.isfinite(patch).all():
        raise ValueError(f"Patch contains non-finite values: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), patch.astype(np.float32), allow_pickle=False)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise IOError(f"Failed to save patch: {output_path}")
