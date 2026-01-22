#!/usr/bin/env python3
"""
Preprocess 3D medical volumes: 1042×1042×[50-200] → patches saved as .nii.gz
Each patch is saved as a separate .nii.gz file.
"""

import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
import multiprocessing as mp

try:
    import nibabel as nib
    from utils import load_volume, select_best_slices, extract_patches, resize_patch, normalize_patch
except ImportError:
    print("ERROR: utils.py not found or nibabel not installed")
    exit(1)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_patch_nii(patch, output_path, affine=None):
    """Save patch as .nii.gz file."""
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


def process_single_volume(args_tuple):
    """
    Process a single volume - designed for multiprocessing
    
    Args:
        args_tuple: Tuple of (stack, data_root, patches_output, cfg)
        
    Returns:
        Tuple of (patches_info_list, success, error_message)
    """
    stack, data_root, patches_output, cfg = args_tuple
    
    stack_id = stack['id']
    
    classe = stack.get('infos', {}).get('Classe', '')
    if classe == 'SAIN':
        label = 0
    elif classe == 'MALADE':
        label = 1
    else:
        return ([], False, f"Invalid class: {classe}")
    
    nii_path = stack.get('nii_path')
    if not nii_path:
        return ([], False, "No nii_path")
    
    # Files are always .nii.gz
    nii_filename = Path(nii_path).name
    vol_path = data_root / nii_filename.replace('.nii', '.nii.gz')
    
    if not vol_path.exists():
        return ([], False, f"File not found: {vol_path}")
    
    try:
        # Load and preprocess volume
        vol = load_volume(vol_path)
        vol = select_best_slices(vol, cfg['target_depth'], cfg['slice_selection'])
        patches = extract_patches(vol, cfg['n_patches_h'], cfg['n_patches_w'])
        
        patches_info_list = []
        
        # Process and save each patch
        for i in range(cfg['n_patches_h']):
            for j in range(cfg['n_patches_w']):
                patch_idx = i * cfg['n_patches_w'] + j
                patch = patches[patch_idx]
                
                # Resize and normalize
                patch = resize_patch(patch, cfg['target_height'], cfg['target_width'])
                patch = normalize_patch(patch, cfg['normalization'])
                
                # Save as .nii.gz
                patch_filename = f"{stack_id}_patch_{i:02d}_{j:02d}.nii.gz"
                patch_path = patches_output / patch_filename
                
                save_patch_nii(patch, patch_path)
                
                patches_info_list.append({
                    'filename': patch_filename,
                    'stack_id': stack_id,
                    'label': label,
                    'position_i': i,
                    'position_j': j
                })
        
        return (patches_info_list, True, None)
        
    except Exception as e:
        return ([], False, f"Error on {stack_id}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    
    # Paths
    paths = config['paths']
    # Handle absolute and relative paths
    dataset_json_path = Path(paths['dataset_json'])
    if dataset_json_path.is_absolute():
        dataset_path = dataset_json_path
    else:
        dataset_path = (config_path.parent.parent / paths['dataset_json']).resolve()
    
    # No split used during preprocessing - all volumes are processed
    data_root = Path(paths['data_root'])
    
    output_dir_path = Path(paths['output_dir'])
    if output_dir_path.is_absolute():
        output_dir = output_dir_path
    else:
        output_dir = (config_path.parent.parent / paths['output_dir']).resolve()
    
    # Config info
    cfg = config['preprocessing']
    n_patches = cfg['n_patches_h'] * cfg['n_patches_w']
    patch_shape = (cfg['target_height'], cfg['target_width'], cfg['target_depth'], 3)
    
    print(f"\nConfig: {cfg['target_height']}x{cfg['target_width']}x{cfg['target_depth']}, "
          f"{cfg['n_patches_h']}x{cfg['n_patches_w']}={n_patches} patches/volume")
    
    # Load dataset (all volumes, no train/test filtering)
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"Data: {len(dataset['stacks'])} total volumes (no train/test split during preprocessing)")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = cfg['target_height']
    w = cfg['target_width']
    d = cfg['target_depth']
    
    # Single output directory for all patches (no train/test separation)
    patches_output = output_dir / f"preprocessed_{h}x{w}x{d}_{timestamp}" / "patches"
    
    print(f"\nOutput: {patches_output.parent}")
    
    # Process volumes in parallel using all CPU cores
    print("\nProcessing...")
    
    # Process all stacks (no filtering)
    stacks_to_process = dataset['stacks']
    total_volumes = len(stacks_to_process)
    
    # Filter valid stacks (SAIN/MALADE with nii_path)
    valid_stacks = []
    for stack in stacks_to_process:
        classe = stack.get('infos', {}).get('Classe', '')
        if classe not in ['SAIN', 'MALADE']:
            continue
        if not stack.get('nii_path'):
            continue
        nii_filename = Path(stack['nii_path']).name
        vol_path = data_root / nii_filename.replace('.nii', '.nii.gz')
        if vol_path.exists():
            valid_stacks.append(stack)
    
    print(f"Valid volumes to process: {len(valid_stacks)}/{total_volumes}")
    
    # Prepare arguments for multiprocessing
    n_workers = mp.cpu_count()
    print(f"Using {n_workers} CPU cores for parallel processing")
    
    process_args = [
        (stack, data_root, patches_output, cfg)
        for stack in valid_stacks
    ]
    
    start_time = time.time()
    patches_info = []
    errors = 0
    volume_count = 0
    
    # Process volumes in parallel
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_volume, process_args),
            total=len(process_args),
            desc="Processing volumes"
        ))
    
    # Collect results
    for patches_info_list, success, error_msg in results:
        if success:
            patches_info.extend(patches_info_list)
            volume_count += 1
        else:
            errors += 1
            if error_msg:
                print(f"Error: {error_msg}")
    
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    
    print(f"\n[{volume_count}/{len(valid_stacks)}] Progression: 100.0% | "
          f"Temps total: {elapsed_min:.1f} min | "
          f"Temps moyen/volume: {elapsed_time/volume_count if volume_count > 0 else 0:.2f}s | "
          f"Volumes traités: {volume_count} | Erreurs: {errors}")
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'config': {
            'target_height': h,
            'target_width': w,
            'target_depth': d,
            'n_patches_h': cfg['n_patches_h'],
            'n_patches_w': cfg['n_patches_w'],
            'n_patches_per_volume': n_patches,
            'slice_selection': cfg['slice_selection'],
            'normalization': cfg['normalization']
        },
        'stats': {
            'total_volumes': volume_count,
            'total_patches': volume_count * n_patches,
            'errors': errors
        }
    }
    
    output_base = output_dir / f"preprocessed_{h}x{w}x{d}_{timestamp}"
    
    # Save metadata
    metadata_path = output_base / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save patches information (stack_id, label, position_i, position_j)
    patches_info_path = output_base / "patches_info.json"
    with open(patches_info_path, 'w') as f:
        json.dump(patches_info, f, indent=2)
    
    print(f"\nDone: {output_dir / f'preprocessed_{h}x{w}x{d}_{timestamp}'}")
    print(f"Total: {volume_count * n_patches:,} patches ({volume_count} volumes)")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
