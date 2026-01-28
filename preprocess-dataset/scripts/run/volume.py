"""Process a single volume: load, normalize, extract patches, save."""

from pathlib import Path
import numpy as np

from ioutils.nii import load_volume, save_patch_nii
from normalize import normalize_patch
from patches import extract_patches_max, extract_patches_top_n
from config.loader import get_slice_selection_method


def process_single_volume(args_tuple) -> tuple:
    """Process one volume. Returns (patches_info_list, success, error_msg)."""
    stack, data_root, patches_output, cfg, norm_config, patch_mode, n_patches, pool_stride = args_tuple

    stack_id = stack["id"]
    classe = stack.get("infos", {}).get("Classe", "")
    if classe == "SAIN":
        label = 0
    elif classe == "MALADE":
        label = 1
    else:
        return ([], False, f"Invalid class: {classe}")

    nii_path = stack.get("nii_path")
    if not nii_path:
        return ([], False, "No nii_path")
    vol_path = data_root / Path(nii_path).name.replace(".nii", ".nii.gz")
    if not vol_path.exists():
        return ([], False, f"File not found: {vol_path}")

    try:
        vol = load_volume(vol_path)
        vol = normalize_patch(
            vol,
            method=norm_config["method"],
            clip_min=norm_config.get("clip_min"),
            clip_max=norm_config.get("clip_max"),
            scale_below_range=norm_config.get("scale_below_range"),
            scale_above_range=norm_config.get("scale_above_range"),
            scale_middle_range=norm_config.get("scale_middle_range"),
        )

        target_h, target_w, target_d = cfg["target_height"], cfg["target_width"], cfg["target_depth"]
        slice_method, min_intensity, max_intensity = get_slice_selection_method(cfg)

        if patch_mode == "max":
            patches, positions = extract_patches_max(vol, target_h, target_w, target_d, slice_method, min_intensity, max_intensity)
        elif patch_mode == "top_n":
            patches, positions = extract_patches_top_n(vol, n_patches, target_h, target_w, target_d, pool_stride, slice_method, min_intensity, max_intensity)
        else:
            return ([], False, f"Unknown patch mode: {patch_mode}")

        if not patches:
            return ([], False, f"No patches extracted from {stack_id}")

        patches_info_list = []
        for patch_idx, (patch, pos) in enumerate(zip(patches, positions)):
            try:
                if patch.size == 0 or not np.isfinite(patch).all():
                    raise ValueError(f"Invalid patch at index {patch_idx}")
                patch_filename = f"{stack_id}_patch_{patch_idx:03d}.nii.gz"
                save_patch_nii(patch, patches_output / patch_filename)
                info = {"filename": patch_filename, "stack_id": stack_id, "label": label, "patch_index": patch_idx}
                if len(pos) == 3:
                    info["position_h"], info["position_w"], info["position_d"] = int(pos[0]), int(pos[1]), int(pos[2])
                else:
                    info["position_h"], info["position_w"] = int(pos[0]), int(pos[1])
                patches_info_list.append(info)
            except Exception as e:
                print(f"  WARNING: Error processing patch {patch_idx} for {stack_id}: {e}")

        return (patches_info_list, True, None)
    except Exception as e:
        return ([], False, f"Error on {stack_id}: {e}")
