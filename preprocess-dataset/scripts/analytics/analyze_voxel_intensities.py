#!/usr/bin/env python3
"""Analyze voxel intensity distribution across all volumes.

Main entry point for voxel intensity analysis.
Processes volumes in parallel without loading everything in RAM.
Computes histograms in workers to avoid sending all voxels through multiprocessing.

See analytics.io for the output JSON format documentation.
"""

import sys
import argparse
from pathlib import Path

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))

from analytics.io import load_volume_paths, save_analysis_data
from analytics.processing import process_volumes_two_pass, compute_distribution_from_histograms
from analytics.stats import compute_percentiles_from_histogram
from analytics.visualization import plot_distribution, plot_statistics


def main():
    parser = argparse.ArgumentParser(description='Analyze voxel intensity distribution')
    parser.add_argument('--dataset-json', type=str, required=True, help='JSON file with volume paths')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory where .nii.gz files are located')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: same as JSON)')
    parser.add_argument('--bins', type=int, default=1000, help='Number of bins (default: 1000)')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: min(8, CPU count) to avoid RAM issues)')
    parser.add_argument('--from-json', type=str, default=None, help='Regenerate plots from JSON file instead of processing volumes')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If --from-json is provided, just regenerate plots
    if args.from_json:
        from analytics.visualization import plot_from_json
        json_path = Path(args.from_json)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        plot_from_json(json_path, output_dir)
        print("\n✅ Plots regenerated from JSON!")
        return
    
    # Normal processing
    dataset_json = Path(args.dataset_json)
    data_root = Path(args.data_root)
    
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    # Load volume paths
    print("Loading volume paths from dataset JSON...")
    volume_paths = load_volume_paths(dataset_json, data_root)
    print(f"Found {len(volume_paths)} volumes")
    
    # Process volumes
    all_stats = process_volumes_two_pass(volume_paths, bins=args.bins, n_workers=args.workers)
    
    # Compute distribution from histograms
    bin_centers, proportions, total_counts, global_stats = compute_distribution_from_histograms(all_stats, bins=args.bins)
    
    # Compute percentiles
    percentiles = compute_percentiles_from_histogram(bin_centers, total_counts, global_stats['total_voxels'])
    
    # Save data to JSON
    metadata = {
        'bins': args.bins,
        'n_volumes': len(all_stats),
        'dataset_json': str(dataset_json),
        'data_root': str(data_root)
    }
    
    json_output_path = output_dir / 'voxel_intensity_analysis.json'
    save_analysis_data(
        json_output_path,
        bin_centers.tolist(),
        proportions.tolist(),
        total_counts.tolist(),
        global_stats,
        percentiles,
        metadata
    )
    
    # Generate plots
    plot_distribution(bin_centers, proportions, output_dir / 'voxel_intensity_distribution.png')
    plot_statistics(bin_centers, total_counts, global_stats['total_voxels'], percentiles, global_stats, output_dir / 'voxel_intensity_statistics.png')
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
