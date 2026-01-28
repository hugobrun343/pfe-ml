"""Sample validation: shapes, norm range, finiteness."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


def check_sample(
    files: List[Path],
    sample_size: int,
    load_patch: Callable[[Path], np.ndarray],
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> List[str]:
    """
    Validate a sample of patches: consistent shapes, expected (H,W,D) or (H,W,D,C), [0,1] norm, finite values.
    expected_shape: optional (H,W,D) or (H,W,D,C); if (H,W,D), checks patch.shape[:3]; if (H,W,D,C), checks full shape.
    Returns a list of issue strings (empty if all pass).
    """
    sample = files[: min(sample_size, len(files))]
    issues = []
    shapes = set()

    for f in sample:
        try:
            patch = load_patch(f)
            sh = patch.shape
            shapes.add(sh)
            if expected_shape is not None:
                if len(expected_shape) == 3:
                    if sh[:3] != expected_shape:
                        issues.append(f"{f.name}: shape {sh}, expected (H,W,D)={expected_shape}")
                else:
                    if sh != expected_shape:
                        issues.append(f"{f.name}: shape {sh}, expected {expected_shape}")
            if not np.isfinite(patch).all():
                issues.append(f"{f.name}: non-finite values")
                continue
            mn, mx = float(np.min(patch)), float(np.max(patch))
            if mn < -0.1 or mx > 1.1:
                issues.append(f"{f.name}: range [{mn:.3f}, {mx:.3f}] (expected ~[0,1])")
        except Exception as e:
            issues.append(f"{f.name}: {e}")

    if expected_shape is None and len(shapes) > 1:
        issues.append(f"Inconsistent shapes in sample: {shapes}")

    return issues
