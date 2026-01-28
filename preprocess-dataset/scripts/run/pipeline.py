"""Pipeline: process_all_volumes (parallel orchestration)."""

import time
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from run.volume import process_single_volume


def process_all_volumes(
    context: dict,
    valid_stacks: list,
    output_base: Path,
    patches_output: Path,
    n_workers: int,
) -> tuple:
    """Process all volumes in parallel. Returns (patches_info, volume_count, errors, elapsed_time, n_patches_per_volume)."""
    cfg = context["cfg"]
    paths = context["paths"]
    norm_config = context["norm_config"]
    patch_mode = context["patch_mode"]
    n_patches = context["n_patches"]
    pool_stride = context["pool_stride"]

    process_args = [
        (s, paths["data_root"], patches_output, cfg, norm_config, patch_mode, n_patches, pool_stride)
        for s in valid_stacks
    ]
    chunksize = max(1, len(process_args) // (n_workers * 4))

    start = time.time()
    patches_info = []
    volume_count = 0
    errors = 0

    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_volume, process_args, chunksize=chunksize),
            total=len(process_args),
            desc="Processing volumes",
        ))

    for info_list, success, error_msg in results:
        if success:
            patches_info.extend(info_list)
            volume_count += 1
        else:
            errors += 1
            if error_msg:
                print(f"Error: {error_msg}")

    elapsed = time.time() - start

    if patch_mode == "top_n":
        n_patches_per_volume = n_patches
    elif patches_info:
        n_patches_per_volume = sum(1 for p in patches_info if p["stack_id"] == patches_info[0]["stack_id"])
    else:
        n_patches_per_volume = (1042 // cfg["target_height"]) * (1042 // cfg["target_width"])

    return patches_info, volume_count, errors, elapsed, n_patches_per_volume
