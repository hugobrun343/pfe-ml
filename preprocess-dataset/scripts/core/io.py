"""I/O operations for loading and saving NIfTI files."""

import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")


def load_volume(path: Path) -> np.ndarray:
    """
    Load NIfTI (.nii.gz) volume and convert to standard format (H, W, D, C).
    
    Args:
        path: Path to .nii.gz file
        
    Returns:
        Volume array with shape (H, W, D, C)
    """
    img = nib.load(str(path))
    vol = img.get_fdata()
    
    # NIfTI format from dataset: (Z, Y, X, C)
    # Convert to (H, W, D, C) = (Y, X, Z, C)
    if vol.ndim == 4:  # (Z, Y, X, C)
        vol = vol.transpose(1, 2, 0, 3)  # → (Y, X, Z, C) = (H, W, D, C)
    elif vol.ndim == 3:  # Grayscale (Z, Y, X)
        vol = vol.transpose(1, 2, 0)  # → (Y, X, Z)
        vol = np.expand_dims(vol, axis=-1)  # → (Y, X, Z, 1)
    
    return vol.astype(np.float32)


def save_patch_nii(patch: np.ndarray, output_path: Path, affine: np.ndarray = None) -> None:
    """
    Save patch as compressed .nii.gz file.
    
    Args:
        patch: Patch array with shape (H, W, D, C)
        output_path: Output file path (.nii.gz)
        affine: Optional affine transformation matrix (default: identity)
    """
    # Convert from (H, W, D, C) to (D, H, W, C) for NIfTI format
    # NIfTI expects (Z, Y, X, C) = (D, H, W, C)
    patch_nii = np.transpose(patch, (2, 0, 1, 3))
    
    # Create NIfTI image
    if affine is None:
        # Default affine matrix (identity)
        affine = np.eye(4)
    
    img = nib.Nifti1Image(patch_nii.astype(np.float32), affine)
    
    # Save compressed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))
