"""Patch validation: orchestration and post-check entrypoint."""

import sys
from pathlib import Path
from typing import Optional, Tuple

from .counts import compute_per_volume_counts, format_missing_volumes_report
from .loader import load_patch_as_hwdc
from .sample_checks import check_sample


def run_validation(
    patches_dir: Path,
    expected_n_volumes: int,
    expected_patches_per_volume: int,
    sample_size: int = 20,
    expected_shape: Optional[Tuple[int, ...]] = None,
    output_format: str = "nii.gz",
) -> Tuple[bool, str]:
    """
    Validate patches folder: count, empty files, per-volume counts, sample (shape, norm ~[0,1], finiteness).
    expected_shape: optional (H,W,D) or (H,W,D,C) for patch size, e.g. (256, 256, 32) or (256, 256, 32, 3).
    output_format: 'nii.gz' or 'npy' - which files to look for.
    Returns (ok, message).
    """
    folder = Path(patches_dir)
    if not folder.exists():
        return False, f"Folder does not exist: {folder}"

    pattern = "*.nii.gz" if output_format == "nii.gz" else "*.npy"
    files = sorted(folder.glob(pattern))
    total = len(files)
    expected = expected_n_volumes * expected_patches_per_volume

    if total == 0:
        return False, f"No {pattern} files found."

    empty = [f.name for f in files if f.stat().st_size == 0]
    if empty:
        return False, f"{len(empty)} empty file(s): " + ", ".join(empty[:5]) + ("..." if len(empty) > 5 else "")

    count_ok = total == expected
    per_volume = compute_per_volume_counts(files)
    sample_issues = check_sample(files, sample_size, load_patch_as_hwdc, expected_shape=expected_shape)

    if sample_issues:
        return False, "Validation issues:\n  " + "\n  ".join(sample_issues[:10])

    if not count_ok:
        detail = format_missing_volumes_report(
            per_volume, expected_patches_per_volume, total, expected
        )
        return True, (
            f"Count mismatch: {total} files, expected {expected}. Sample checks passed.\n"
            f"Volumes with missing patches:\n{detail}"
        )

    return True, f"OK: {total} patches, sample checks passed."


def run_post_check(
    patches_dir: Path,
    expected_n_volumes: int,
    expected_patches_per_volume: int,
    expected_shape: Optional[Tuple[int, ...]] = None,
    output_format: str = "nii.gz",
) -> None:
    """
    Run run_validation, print the message, and exit(1) if validation fails.
    expected_shape: optional (H,W,D) or (H,W,D,C), e.g. (256, 256, 32, 3).
    output_format: 'nii.gz' or 'npy'.
    """
    ok, msg = run_validation(
        patches_dir, expected_n_volumes, expected_patches_per_volume,
        expected_shape=expected_shape, output_format=output_format,
    )
    print(f"\n[Check] {msg}")
    if not ok:
        sys.exit(1)
