#!/usr/bin/env python3
"""Analyze training results to identify problematic positions and volumes"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys

# Add parent directory to path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(json_path: Path) -> dict:
    """Load training results JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_positions(results: dict) -> Dict[Tuple[int, int], dict]:
    """
    Analyze failure rate by grid position
    
    Returns:
        Dict mapping (i, j) -> {'total': int, 'failures': int, 'failure_rate': float}
    """
    position_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    # Use the last epoch for analysis (best model)
    last_epoch = results['epochs'][-1]
    if 'validation' not in last_epoch or 'samples' not in last_epoch['validation']:
        return {}
    
    samples = last_epoch['validation']['samples']
    
    for sample in samples:
        if 'grid_position' not in sample:
            continue
        
        i, j = sample['grid_position']
        position_stats[(i, j)]['total'] += 1
        if not sample['correct']:
            position_stats[(i, j)]['failures'] += 1
    
    # Calculate failure rates
    position_analysis = {}
    for pos, stats in position_stats.items():
        position_analysis[pos] = {
            'total': stats['total'],
            'failures': stats['failures'],
            'failure_rate': stats['failures'] / stats['total'] if stats['total'] > 0 else 0.0
        }
    
    return position_analysis


def analyze_volumes(results: dict) -> Dict[str, dict]:
    """
    Analyze failure rate by volume (stack_id)
    
    Returns:
        Dict mapping stack_id -> {'total': int, 'failures': int, 'failure_rate': float}
    """
    volume_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    # Use the last epoch for analysis
    last_epoch = results['epochs'][-1]
    if 'validation' not in last_epoch or 'samples' not in last_epoch['validation']:
        return {}
    
    samples = last_epoch['validation']['samples']
    
    for sample in samples:
        if 'stack_id' not in sample:
            continue
        
        stack_id = sample['stack_id']
        volume_stats[stack_id]['total'] += 1
        if not sample['correct']:
            volume_stats[stack_id]['failures'] += 1
    
    # Calculate failure rates
    volume_analysis = {}
    for stack_id, stats in volume_stats.items():
        volume_analysis[stack_id] = {
            'total': stats['total'],
            'failures': stats['failures'],
            'failure_rate': stats['failures'] / stats['total'] if stats['total'] > 0 else 0.0
        }
    
    return volume_analysis


def analyze_combined(results: dict) -> Dict[Tuple[str, Tuple[int, int]], dict]:
    """
    Analyze failure rate by (volume, position) combination
    
    Returns:
        Dict mapping (stack_id, (i, j)) -> {'total': int, 'failures': int, 'failure_rate': float}
    """
    combined_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    # Use the last epoch for analysis
    last_epoch = results['epochs'][-1]
    if 'validation' not in last_epoch or 'samples' not in last_epoch['validation']:
        return {}
    
    samples = last_epoch['validation']['samples']
    
    for sample in samples:
        if 'stack_id' not in sample or 'grid_position' not in sample:
            continue
        
        stack_id = sample['stack_id']
        i, j = sample['grid_position']
        key = (stack_id, (i, j))
        
        combined_stats[key]['total'] += 1
        if not sample['correct']:
            combined_stats[key]['failures'] += 1
    
    # Calculate failure rates
    combined_analysis = {}
    for key, stats in combined_stats.items():
        combined_analysis[key] = {
            'total': stats['total'],
            'failures': stats['failures'],
            'failure_rate': stats['failures'] / stats['total'] if stats['total'] > 0 else 0.0
        }
    
    return combined_analysis


def plot_position_heatmap(position_analysis: Dict[Tuple[int, int], dict], output_dir: Path):
    """Create heatmap of failure rates by position"""
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


def plot_position_bar_chart(position_analysis: Dict[Tuple[int, int], dict], output_dir: Path):
    """Create bar chart of worst positions"""
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


def plot_volume_analysis(volume_analysis: Dict[str, dict], output_dir: Path):
    """Create bar chart of worst volumes"""
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


def plot_combined_analysis(combined_analysis: Dict[Tuple[str, Tuple[int, int]], dict], output_dir: Path):
    """Create analysis of worst (volume, position) combinations"""
    if not combined_analysis:
        return
    
    # Sort by failure rate
    sorted_combined = sorted(
        combined_analysis.items(),
        key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
        reverse=True
    )
    
    # Take top 30 worst
    top_n = min(30, len(sorted_combined))
    top_combined = sorted_combined[:top_n]
    
    labels = [f"{vol_id}\n({i},{j})" for (vol_id, (i, j)), _ in top_combined]
    failure_rates = [stats['failure_rate'] for _, stats in top_combined]
    totals = [stats['total'] for _, stats in top_combined]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(labels, failure_rates, color='indianred')
    ax.set_xlabel('Failure Rate', fontsize=12)
    ax.set_ylabel('Volume ID & Position', fontsize=12)
    ax.set_title(f'Top {top_n} Worst Performing (Volume, Position) Combinations', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add count annotations
    for i, (bar, total) in enumerate(zip(bars, totals)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'n={total}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_text_report(position_analysis: Dict, volume_analysis: Dict, 
                     combined_analysis: Dict, output_dir: Path):
    """Save text report with detailed statistics"""
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RESULTS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Position analysis
        f.write("WORST PERFORMING GRID POSITIONS\n")
        f.write("-" * 80 + "\n")
        if position_analysis:
            sorted_positions = sorted(
                position_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Position':<15} {'Total':<10} {'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for (i, j), stats in sorted_positions[:20]:
                f.write(f"({i:2d},{j:2d}):{'':<6} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        else:
            f.write("No position data available.\n")
        f.write("\n\n")
        
        # Volume analysis
        f.write("WORST PERFORMING VOLUMES\n")
        f.write("-" * 80 + "\n")
        if volume_analysis:
            sorted_volumes = sorted(
                volume_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Volume ID':<20} {'Total':<10} {'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for vol_id, stats in sorted_volumes[:30]:
                f.write(f"{vol_id:<20} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        else:
            f.write("No volume data available.\n")
        f.write("\n\n")
        
        # Combined analysis
        f.write("WORST PERFORMING (VOLUME, POSITION) COMBINATIONS\n")
        f.write("-" * 80 + "\n")
        if combined_analysis:
            sorted_combined = sorted(
                combined_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Volume ID':<20} {'Position':<15} {'Total':<10} "
                   f"{'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for (vol_id, (i, j)), stats in sorted_combined[:30]:
                f.write(f"{vol_id:<20} ({i:2d},{j:2d}):{'':<6} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        else:
            f.write("No combined data available.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training results to identify problematic positions and volumes'
    )
    parser.add_argument(
        'results_json',
        type=str,
        help='Path to training_results.json file'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    # Output directory is a subdirectory next to results_json
    output_dir = results_path.parent / 'analytics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    print("Analyzing positions...")
    position_analysis = analyze_positions(results)
    
    print("Analyzing volumes...")
    volume_analysis = analyze_volumes(results)
    
    print("Analyzing combined (volume, position)...")
    combined_analysis = analyze_combined(results)
    
    print("Generating visualizations...")
    plot_position_heatmap(position_analysis, output_dir)
    plot_position_bar_chart(position_analysis, output_dir)
    plot_volume_analysis(volume_analysis, output_dir)
    
    print("Saving text report...")
    save_text_report(position_analysis, volume_analysis, combined_analysis, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("Generated files:")
    print("  - analytics/position_analysis_heatmap.png")
    print("  - analytics/position_analysis_barchart.png")
    print("  - analytics/volume_analysis.png")
    print("  - analytics/analysis_report.txt")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
