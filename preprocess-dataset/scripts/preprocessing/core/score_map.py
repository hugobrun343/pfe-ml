"""Score map computation for patch selection."""

import numpy as np
from typing import Literal


def compute_score_map(
    vol: np.ndarray,
    method: Literal['intensity', 'variance', 'entropy', 'gradient']
) -> np.ndarray:
    """
    Compute a 2D score map (H, W) by projecting the 3D volume.
    
    Args:
        vol: Volume array with shape (H, W, D, C)
        method: Scoring method ('intensity', 'variance', 'entropy', 'gradient')
        
    Returns:
        2D score map with shape (H, W)
    """
    H, W, D, C = vol.shape
    
    if method == 'intensity':
        # Mean intensity across all slices for each position
        score_map = np.mean(vol, axis=2)  # (H, W, C)
        if C > 1:
            score_map = np.mean(score_map, axis=2)  # (H, W)
        else:
            score_map = score_map[:, :, 0]  # (H, W)
        return score_map
    
    elif method == 'variance':
        # Variance across slices for each position
        score_map = np.var(vol, axis=2)  # (H, W, C)
        if C > 1:
            score_map = np.mean(score_map, axis=2)  # (H, W)
        else:
            score_map = score_map[:, :, 0]  # (H, W)
        return score_map
    
    elif method == 'entropy':
        # Entropy across slices for each position
        score_map = np.zeros((H, W), dtype=np.float32)
        for h in range(H):
            for w in range(W):
                # Compute entropy from histogram of values across slices
                values = vol[h, w, :, :].flatten()
                hist, _ = np.histogram(values, bins=256)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    hist = hist / hist.sum()
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    score_map[h, w] = entropy
        return score_map
    
    elif method == 'gradient':
        # Gradient magnitude averaged across slices
        score_map = np.zeros((H, W), dtype=np.float32)
        for d in range(D):
            for c in range(C):
                slice_2d = vol[:, :, d, c]
                # Compute gradient magnitude
                grad_h = np.abs(np.diff(slice_2d, axis=0, prepend=slice_2d[0:1, :]))
                grad_w = np.abs(np.diff(slice_2d, axis=1, prepend=slice_2d[:, 0:1]))
                grad_mag = np.sqrt(grad_h**2 + grad_w**2)
                score_map += grad_mag
        score_map /= (D * C)  # Average across slices and channels
        return score_map
    
    else:
        raise ValueError(f"Unknown scoring method: {method}")
