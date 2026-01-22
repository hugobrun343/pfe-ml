#!/usr/bin/env python3
"""
Preprocess 3D medical volumes: 1042×1042×[50-200] → patches saved as .nii.gz

Each patch is saved as a separate .nii.gz file.
All volumes are preprocessed without train/test split (splits are applied during data loading).
"""

import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import time
import multiprocessing as mp

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))
sys.path.insert(0, str(scripts_dir / "helpers"))

# Core imports (fundamental preprocessing functions)
import core.io as nii_io
import core.utils
import core.slice_selection
import core.normalization

# Helper imports (utilities and orchestration)
import helpers.config_loader
import helpers.processing
import helpers.metadata
import helpers.display

# Import functions from core
load_volume = nii_io.load_volume
save_patch_nii = nii_io.save_patch_nii
extract_patches = core.utils.extract_patches
resize_patch = core.utils.resize_patch
select_best_slices = core.slice_selection.select_best_slices
normalize_patch = core.normalization.normalize_patch
get_normalization_config = core.normalization.get_normalization_config

# Import functions from helpers
load_config = helpers.config_loader.load_config
resolve_paths = helpers.config_loader.resolve_paths
get_slice_selection_method = helpers.config_loader.get_slice_selection_method
filter_valid_stacks = helpers.processing.filter_valid_stacks
compute_global_normalization_stats = helpers.processing.compute_global_normalization_stats
compute_sample_normalization_stats = helpers.processing.compute_sample_normalization_stats
build_metadata = helpers.metadata.build_metadata
print_config_summary = helpers.display.print_config_summary


def process_single_volume(args_tuple):
    """
    Process a single volume - designed for multiprocessing.
    
    Args:
        args_tuple: Tuple of (stack, data_root, patches_output, cfg, norm_config, norm_stats)
        
    Returns:
        Tuple of (patches_info_list, success, error_message)
    """
    stack, data_root, patches_output, cfg, norm_config, norm_stats = args_tuple
    
    stack_id = stack['id']
    
    # Extract label from class
    classe = stack.get('infos', {}).get('Classe', '')
    if classe == 'SAIN':
        label = 0
    elif classe == 'MALADE':
        label = 1
    else:
        return ([], False, f"Invalid class: {classe}")
    
    # Get volume path
    nii_path = stack.get('nii_path')
    if not nii_path:
        return ([], False, "No nii_path")
    
    nii_filename = Path(nii_path).name
    vol_path = data_root / nii_filename.replace('.nii', '.nii.gz')
    
    if not vol_path.exists():
        return ([], False, f"File not found: {vol_path}")
    
    try:
        # Load and preprocess volume
        vol = load_volume(vol_path)
        
        # Get slice selection method
        slice_method = get_slice_selection_method(cfg)
        vol = select_best_slices(vol, cfg['target_depth'], slice_method)
        patches = extract_patches(vol, cfg['n_patches_h'], cfg['n_patches_w'])
        
        patches_info_list = []
        
        # Process and save each patch
        for i in range(cfg['n_patches_h']):
            for j in range(cfg['n_patches_w']):
                patch_idx = i * cfg['n_patches_w'] + j
                patch = patches[patch_idx]
                
                # Resize
                patch = resize_patch(patch, cfg['target_height'], cfg['target_width'])
                
                # Normalize using modular normalization
                patch = normalize_patch(
                    patch,
                    method=norm_config['method'],
                    clip_min=norm_config.get('clip_min'),
                    clip_max=norm_config.get('clip_max'),
                    below_clip_value=norm_config.get('below_clip_value'),
                    above_clip_value=norm_config.get('above_clip_value'),
                    scale_below_range=norm_config.get('scale_below_range'),
                    scale_above_range=norm_config.get('scale_above_range'),
                    scale_middle_range=norm_config.get('scale_middle_range'),
                    stats=norm_stats
                )
                
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
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess 3D medical volumes into patches")
    parser.add_argument('--config', '-c', required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load and validate configuration
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    
    # Extract configuration sections
    cfg = config['preprocessing']
    version = config['version']
    norm_config = get_normalization_config(cfg)
    
    # Print configuration summary
    print_config_summary(cfg, norm_config, version)
    
    # Load dataset
    with open(paths['dataset_json'], 'r') as f:
        dataset = json.load(f)
    print(f"\nData: {len(dataset['stacks'])} total volumes (no train/test split during preprocessing)")
    
    # Filter valid stacks
    valid_stacks = filter_valid_stacks(dataset, paths['data_root'])
    print(f"Valid volumes to process: {len(valid_stacks)}/{len(dataset['stacks'])}")
    
    # Compute normalization statistics if needed
    norm_stats = None
    if norm_config.get('normalize_globally', False):
        norm_stats = compute_global_normalization_stats(valid_stacks, paths['data_root'], cfg, norm_config)
    elif norm_config.get('compute_on_sample', False) and norm_config['method'] in ['percentile', 'robust']:
        norm_stats = compute_sample_normalization_stats(valid_stacks, paths['data_root'], cfg, norm_config)
    
    # Create output directories
    h, w, d = cfg['target_height'], cfg['target_width'], cfg['target_depth']
    base_name = f"preprocessed_{h}x{w}x{d}"
    output_base = paths['output_dir'] / f"{base_name}_{version}"
    patches_output = output_base / "patches"
    patches_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput: {output_base}")
    
    # Process volumes in parallel
    print("\nProcessing...")
    n_workers = mp.cpu_count()
    print(f"Using {n_workers} CPU cores for parallel processing")
    
    process_args = [
        (stack, paths['data_root'], patches_output, cfg, norm_config, norm_stats)
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
    
    # Build and save metadata
    n_patches = cfg['n_patches_h'] * cfg['n_patches_w']
    metadata = build_metadata(
        config, version, paths['dataset_json'], cfg, norm_config,
        n_workers, elapsed_time, volume_count, n_patches, errors, norm_stats
    )
    
    metadata_path = output_base / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save patches information
    patches_info_path = output_base / "patches_info.json"
    with open(patches_info_path, 'w') as f:
        json.dump(patches_info, f, indent=2)
    
    print(f"\nDone: {output_base}")
    print(f"Version: {version}")
    print(f"Total: {volume_count * n_patches:,} patches ({volume_count} volumes)")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
