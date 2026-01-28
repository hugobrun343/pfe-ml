#!/usr/bin/env python3
"""Entrypoint: voxel intensity analysis."""

import argparse
import sys
from pathlib import Path

# Run from voxel-analytics/: python run.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.dataset_io import load_volume_paths, save_analysis_data, load_analysis_data
from scripts.processing import process_volumes_two_pass, compute_distribution_from_histograms
from scripts.stats import compute_percentiles_from_histogram
from scripts.visualization import plot_distribution, plot_statistics, plot_from_json


def main():
    p = argparse.ArgumentParser(description="Analyze voxel intensity distribution")
    p.add_argument("--dataset-json", type=str, help="JSON dataset (stacks, nii_path)")
    p.add_argument("--data-root", type=str, help="Root folder for .nii.gz files")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--bins", type=int, default=1000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--from-json", type=str, help="Replot from this JSON file")
    args = p.parse_args()

    if args.from_json:
        out = Path(args.output_dir or Path(args.from_json).parent)
        out.mkdir(parents=True, exist_ok=True)
        plot_from_json(Path(args.from_json), out)
        return

    if not args.dataset_json or not args.data_root:
        p.error("--dataset-json and --data-root required (except with --from-json)")
    dataset_json = Path(args.dataset_json)
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir) if args.output_dir else dataset_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_volume_paths(dataset_json, data_root)
    print(f"Found {len(paths)} volumes")
    all_stats = process_volumes_two_pass(paths, bins=args.bins, n_workers=args.workers)
    bin_centers, proportions, total_counts, global_stats = compute_distribution_from_histograms(
        all_stats, bins=args.bins
    )
    percentiles = compute_percentiles_from_histogram(
        bin_centers, total_counts, global_stats["total_voxels"]
    )
    meta = {"bins": args.bins, "n_volumes": len(all_stats), "dataset_json": str(dataset_json), "data_root": str(data_root)}
    json_path = out_dir / "voxel_intensity_analysis.json"
    save_analysis_data(
        json_path,
        bin_centers.tolist(),
        proportions.tolist(),
        total_counts.tolist(),
        global_stats,
        percentiles,
        meta,
    )
    plot_distribution(bin_centers, proportions, out_dir / "voxel_intensity_distribution.png")
    plot_statistics(
        bin_centers, total_counts, global_stats["total_voxels"],
        percentiles, global_stats,
        out_dir / "voxel_intensity_statistics.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
