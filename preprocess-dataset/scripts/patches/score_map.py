"""Score volumes for patch selection (2D for max, 3D for top_n)."""

import numpy as np
import torch
import torch.nn.functional as F


def compute_score_volume_3d(vol: np.ndarray, pool_stride: int = 2) -> np.ndarray:
    """Max-pool 3D on volume (H,W,D,C) → grid (H',W',D') for top_n."""
    v = np.mean(vol, axis=-1).astype(np.float32)  # (H,W,D)
    t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    if pool_stride > 1:
        pooled = F.max_pool3d(t, kernel_size=pool_stride, stride=pool_stride)
        out = pooled.squeeze(0).squeeze(0).numpy()
    else:
        out = v
    return out
