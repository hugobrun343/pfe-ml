"""Visualization functions for training results analysis."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def plot_position_heatmap(position_analysis: Dict[Tuple[int, int], dict], output_dir: Path) -> None:
    """
    Create heatmap of failure rates by position.
    
    Args:
        position_analysis: Dictionary mapping (i, j) -> stats
        output_dir: Directory to save the plot
    """
    if not position_analysis:
        return
    
    # Find max i and j
    max_i = max(pos[0] for pos in position_analysis.keys())
    max_j = max(pos[1] for pos in position_analysis.keys())
    
    # Create matrix
    heatmap_data = np.full((max_i + 1, max_j + 1), np.nan)
    count_data = np.zeros((max_i + 1, max_j + 1), dtype=int)
    
    for (i, j), stats in position_analysis.items():
        heatmap_data[i, j] = stats['failure_rate']
        count_data[i, j] = stats['total']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap of failure rates
    sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='Reds', 
                cbar_kws={'label': 'Failure Rate'}, ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Failure Rate by Grid Position', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position J', fontsize=12)
    ax1.set_ylabel('Position I', fontsize=12)
    
    # Heatmap of sample counts
    sns.heatmap(count_data, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Sample Count'}, ax=ax2)
    ax2.set_title('Sample Count by Grid Position', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Position J', fontsize=12)
    ax2.set_ylabel('Position I', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_analysis_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_position_bar_chart(position_analysis: Dict[Tuple[int, int], dict], output_dir: Path) -> None:
    """
    Create bar chart of worst positions.
    
    Args:
        position_analysis: Dictionary mapping (i, j) -> stats
        output_dir: Directory to save the plot
    """
    if not position_analysis:
        return
    
    # Sort by failure rate
    sorted_positions = sorted(
        position_analysis.items(),
        key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
        reverse=True
    )
    
    # Take top 20 worst
    top_n = min(20, len(sorted_positions))
    top_positions = sorted_positions[:top_n]
    
    positions_str = [f"({i},{j})" for (i, j), _ in top_positions]
    failure_rates = [stats['failure_rate'] for _, stats in top_positions]
    totals = [stats['total'] for _, stats in top_positions]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(positions_str, failure_rates, color='coral')
    ax.set_xlabel('Failure Rate', fontsize=12)
    ax.set_ylabel('Grid Position (I, J)', fontsize=12)
    ax.set_title(f'Top {top_n} Worst Performing Grid Positions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add count annotations
    for i, (bar, total) in enumerate(zip(bars, totals)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'n={total}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_analysis_barchart.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_volume_analysis(volume_analysis: Dict[str, dict], output_dir: Path) -> None:
    """
    Create bar chart of worst volumes.
    
    Args:
        volume_analysis: Dictionary mapping stack_id -> stats
        output_dir: Directory to save the plot
    """
    if not volume_analysis:
        return
    
    # Sort by failure rate
    sorted_volumes = sorted(
        volume_analysis.items(),
        key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
        reverse=True
    )
    
    # Take top 30 worst
    top_n = min(30, len(sorted_volumes))
    top_volumes = sorted_volumes[:top_n]
    
    volume_ids = [vol_id for vol_id, _ in top_volumes]
    failure_rates = [stats['failure_rate'] for _, stats in top_volumes]
    totals = [stats['total'] for _, stats in top_volumes]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(volume_ids, failure_rates, color='salmon')
    ax.set_xlabel('Failure Rate', fontsize=12)
    ax.set_ylabel('Volume ID', fontsize=12)
    ax.set_title(f'Top {top_n} Worst Performing Volumes', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add count annotations
    for i, (bar, total) in enumerate(zip(bars, totals)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'n={total}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_patch_index_analysis(patch_index_analysis: Dict[int, dict], output_dir: Path) -> None:
    """
    Create bar chart of failure rates by patch index (for mode 'top_n').
    
    Args:
        patch_index_analysis: Dictionary mapping patch_index -> stats
        output_dir: Directory to save the plot
    """
    if not patch_index_analysis:
        return
    
    # Sort by failure rate
    sorted_indices = sorted(
        patch_index_analysis.items(),
        key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
        reverse=True
    )
    
    # Take top 30 worst
    top_n = min(30, len(sorted_indices))
    top_indices = sorted_indices[:top_n]
    
    patch_indices = [f"Patch {idx}" for idx, _ in top_indices]
    failure_rates = [stats['failure_rate'] for _, stats in top_indices]
    totals = [stats['total'] for _, stats in top_indices]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(patch_indices, failure_rates, color='coral')
    ax.set_xlabel('Failure Rate', fontsize=12)
    ax.set_ylabel('Patch Index', fontsize=12)
    ax.set_title(f'Top {top_n} Worst Performing Patches (by index)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add count annotations
    for i, (bar, total) in enumerate(zip(bars, totals)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'n={total}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'patch_index_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
