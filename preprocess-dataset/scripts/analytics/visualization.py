"""Visualization functions for voxel intensity analysis."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_distribution(bin_centers: np.ndarray, proportions: np.ndarray, output_path: Path) -> None:
    """Plot intensity distribution (linear and log scale).
    
    Args:
        bin_centers: Array of bin center values
        proportions: Array of proportions per bin
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Linear scale
    axes[0].plot(bin_centers, proportions, linewidth=1.5, color='steelblue')
    axes[0].set_xlabel('Voxel Intensity', fontsize=12)
    axes[0].set_ylabel('Proportion of Voxels', fontsize=12)
    axes[0].set_title('Voxel Intensity Distribution (Linear Scale)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(bin_centers, proportions, alpha=0.3, color='steelblue')
    
    # Log scale
    axes[1].plot(bin_centers, proportions, linewidth=1.5, color='coral')
    axes[1].set_xlabel('Voxel Intensity', fontsize=12)
    axes[1].set_ylabel('Proportion of Voxels (log scale)', fontsize=12)
    axes[1].set_title('Voxel Intensity Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(bin_centers, proportions, alpha=0.3, color='coral')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_statistics(
    bin_centers: np.ndarray,
    total_counts: np.ndarray,
    total_voxels: int,
    percentiles: Dict[str, float],
    global_stats: Dict[str, float],
    output_path: Path
) -> None:
    """Plot additional statistics: percentiles and cumulative distribution.
    
    Args:
        bin_centers: Array of bin center values
        total_counts: Array of counts per bin
        total_voxels: Total number of voxels
        percentiles: Dictionary mapping percentile names to values
        global_stats: Dictionary with global statistics
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Percentiles
    percentile_names = ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']
    percentile_labels = ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%']
    percentile_values = [percentiles.get(name, 0) for name in percentile_names]
    
    axes[0].bar(range(len(percentiles)), percentile_values, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(percentiles)))
    axes[0].set_xticklabels(percentile_labels)
    axes[0].set_xlabel('Percentile', fontsize=12)
    axes[0].set_ylabel('Intensity Value', fontsize=12)
    axes[0].set_title('Intensity Percentiles', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, (label, val) in enumerate(zip(percentile_labels, percentile_values)):
        axes[0].text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Right plot: Cumulative Distribution Function (CDF)
    cumulative = np.cumsum(total_counts)
    cumulative_pct = cumulative / total_voxels
    
    axes[1].plot(bin_centers, cumulative_pct, linewidth=2, color='coral')
    axes[1].set_xlabel('Voxel Intensity', fontsize=12)
    axes[1].set_ylabel('Cumulative Proportion', fontsize=12)
    axes[1].set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(bin_centers, cumulative_pct, alpha=0.3, color='coral')
    
    # Add vertical lines for key percentiles
    for p_name, p_value in [('p25', percentiles.get('p25')), ('p50', percentiles.get('p50')), ('p75', percentiles.get('p75'))]:
        if p_value is not None:
            axes[1].axvline(p_value, color='steelblue', linestyle='--', alpha=0.7, linewidth=1.5)
            # Find corresponding y value
            idx = np.searchsorted(bin_centers, p_value)
            if idx < len(cumulative_pct):
                y_val = cumulative_pct[idx]
                axes[1].plot(p_value, y_val, 'o', color='steelblue', markersize=8)
                axes[1].text(p_value, y_val + 0.05, f'{p_name[1:]}%', ha='center', fontsize=10, color='steelblue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Statistics plot saved to: {output_path}")
    plt.close()


def plot_from_json(json_path: Path, output_dir: Path) -> None:
    """Load data from JSON and regenerate plots.
    
    Args:
        json_path: Path to JSON file with analysis data
        output_dir: Directory to save plots
    """
    from .io import load_analysis_data
    
    data = load_analysis_data(json_path)
    
    bin_centers = np.array(data['histogram']['bin_centers'])
    proportions = np.array(data['histogram']['proportions'])
    total_counts = np.array(data['histogram']['counts'])
    global_stats = data['global_stats']
    percentiles = data['percentiles']
    total_voxels = global_stats['total_voxels']
    
    # Regenerate plots
    plot_distribution(bin_centers, proportions, output_dir / 'voxel_intensity_distribution.png')
    plot_statistics(bin_centers, total_counts, total_voxels, percentiles, global_stats, output_dir / 'voxel_intensity_statistics.png')
    
    print("\n✅ Plots regenerated from JSON!")
