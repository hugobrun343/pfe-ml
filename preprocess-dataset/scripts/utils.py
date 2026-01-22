"""Utility functions for volume preprocessing."""

import numpy as np
from skimage.transform import resize

try:
    import nibabel as nib
except ImportError:
    print("ERROR: pip install nibabel")
    exit(1)


def load_volume(path):
    """Load NIfTI (.nii.gz) and convert to (H, W, D, C)."""
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


def select_best_slices(vol, n_slices, method='intensity'):
    """Select best contiguous block of n_slices."""
    H, W, D, C = vol.shape
    
    if D <= n_slices:
        if D < n_slices:
            pad = ((0, 0), (0, 0), (0, n_slices - D), (0, 0))
            vol = np.pad(vol, pad, mode='constant')
        return vol
    
    best_score = -np.inf
    best_start = 0
    
    for start in range(D - n_slices + 1):
        block = vol[:, :, start:start + n_slices, :]
        
        if method == 'intensity':
            score = np.sum(block)
        elif method == 'variance':
            score = np.var(block)
        elif method == 'entropy':
            hist, _ = np.histogram(block.flatten(), bins=256)
            hist = hist[hist > 0] / hist.sum()
            score = -np.sum(hist * np.log2(hist))
        elif method == 'gradient':
            score = np.abs(np.diff(block, axis=0)).sum() + np.abs(np.diff(block, axis=1)).sum()
        
        if score > best_score:
            best_score = score
            best_start = start
    
    return vol[:, :, best_start:best_start + n_slices, :]


def extract_patches(vol, n_h, n_w):
    """Extract patches in grid."""
    H, W, D, C = vol.shape
    patch_h, patch_w = H // n_h, W // n_w
    
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            h_start = i * patch_h
            h_end = (i + 1) * patch_h if i < n_h - 1 else H
            w_start = j * patch_w
            w_end = (j + 1) * patch_w if j < n_w - 1 else W
            patches.append(vol[h_start:h_end, w_start:w_end, :, :])
    
    return patches


def resize_patch(patch, target_h, target_w):
    """Resize spatial dimensions."""
    H, W, D, C = patch.shape
    resized = np.zeros((target_h, target_w, D, C), dtype=patch.dtype)
    
    for d in range(D):
        for c in range(C):
            resized[:, :, d, c] = resize(
                patch[:, :, d, c],
                (target_h, target_w),
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
    
    return resized


def normalize_patch(patch, method='z-score'):
    """Normalize intensities."""
    patch = patch.astype(np.float32)
    
    if method == 'z-score':
        mean, std = np.mean(patch), np.std(patch)
        if std > 0:
            patch = (patch - mean) / std
    elif method == 'min-max':
        vmin, vmax = np.min(patch), np.max(patch)
        if vmax > vmin:
            patch = (patch - vmin) / (vmax - vmin)
    
    return patch


def process_volume(path, config):
    """Full pipeline: Load -> Select -> Extract -> Resize -> Normalize."""
    cfg = config['preprocessing']
    
    vol = load_volume(path)
    vol = select_best_slices(vol, cfg['target_depth'], cfg['slice_selection'])
    patches = extract_patches(vol, cfg['n_patches_h'], cfg['n_patches_w'])
    
    processed = []
    for patch in patches:
        patch = resize_patch(patch, cfg['target_height'], cfg['target_width'])
        patch = normalize_patch(patch, cfg['normalization'])
        processed.append(patch)
    
    return processed
