"""Per-volume patch counts and missing-volumes reporting."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def compute_per_volume_counts(files: List[Path]) -> Dict[str, int]:
    """Count patches per stack_id from filenames (stack_XXXXXX_patch_YYY.nii.gz)."""
    per_volume = defaultdict(int)
    for f in files:
        name = f.stem
        if "_patch_" in name:
            stack_id = name.split("_patch_")[0]
            per_volume[stack_id] += 1
    return dict(per_volume)


def format_missing_volumes_report(
    per_volume: Dict[str, int],
    expected_patches_per_volume: int,
    total: int,
    expected_total: int,
    max_lines: int = 20,
) -> str:
    """Build the 'Volumes with missing patches' detail string."""
    missing_volumes = [
        (sid, expected_patches_per_volume - cnt)
        for sid, cnt in per_volume.items()
        if cnt < expected_patches_per_volume
    ]
    missing_volumes.sort(key=lambda x: -x[1])
    lines = [
        f"  {sid}: {expected_patches_per_volume - n_missing} patches (missing {n_missing})"
        for sid, n_missing in missing_volumes[:max_lines]
    ]
    if len(missing_volumes) > max_lines:
        lines.append(f"  ... and {len(missing_volumes) - max_lines} more volumes with missing patches")
    return "\n".join(lines) if lines else ""
