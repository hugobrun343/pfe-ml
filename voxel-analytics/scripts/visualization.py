"""Plots for distribution and CDF."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

from scripts.dataset_io import load_analysis_data


def plot_distribution(bin_centers: np.ndarray, proportions: np.ndarray, output_path: Path) -> None:
    """Linear and log distribution."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].plot(bin_centers, proportions, linewidth=1.5, color="steelblue")
    axes[0].set_xlabel("Voxel Intensity")
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Distribution (linear)")
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(bin_centers, proportions, alpha=0.3, color="steelblue")
    axes[1].plot(bin_centers, proportions, linewidth=1.5, color="coral")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Voxel Intensity")
    axes[1].set_ylabel("Proportion (log)")
    axes[1].set_title("Distribution (log)")
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(bin_centers, proportions, alpha=0.3, color="coral")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_statistics(
    bin_centers: np.ndarray,
    total_counts: np.ndarray,
    total_voxels: int,
    percentiles: Dict[str, float],
    global_stats: Dict,
    output_path: Path,
) -> None:
    """Percentiles and CDF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    labels = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
    vals = [percentiles.get(n, 0) for n in names]
    axes[0].bar(range(len(vals)), vals, color="steelblue", alpha=0.7)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Percentile")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Percentiles")
    axes[0].grid(True, alpha=0.3, axis="y")
    cum = np.cumsum(total_counts) / total_voxels
    axes[1].plot(bin_centers, cum, linewidth=2, color="coral")
    axes[1].set_xlabel("Voxel Intensity")
    axes[1].set_ylabel("Cumulative proportion")
    axes[1].set_title("CDF")
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(bin_centers, cum, alpha=0.3, color="coral")
    for pn, pv in [("p25", percentiles.get("p25")), ("p50", percentiles.get("p50")), ("p75", percentiles.get("p75"))]:
        if pv is not None:
            axes[1].axvline(pv, color="steelblue", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Statistics plot saved: {output_path}")


def plot_from_json(json_path: Path, output_dir: Path) -> None:
    """Replot from analysis JSON."""
    data = load_analysis_data(json_path)
    bc = np.array(data["histogram"]["bin_centers"])
    prop = np.array(data["histogram"]["proportions"])
    cnt = np.array(data["histogram"]["counts"])
    gs = data["global_stats"]
    perc = data["percentiles"]
    n = gs["total_voxels"]
    plot_distribution(bc, prop, output_dir / "voxel_intensity_distribution.png")
    plot_statistics(bc, cnt, n, perc, gs, output_dir / "voxel_intensity_statistics.png")
    print("Plots regenerated from JSON.")
