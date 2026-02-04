"""
Visualizations - Create plots and charts from results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def setup_style():
    """Setup plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def plot_vram_distribution(df: pd.DataFrame, output_file: Optional[str] = None):
    """
    Plot VRAM distribution for successful tests
    
    Args:
        df: DataFrame with results
        output_file: Optional path to save figure
    """
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        print("No successful tests to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram
    axes[0, 0].hist(successful['vram_peak_gb'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('VRAM Peak (GB)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of VRAM Peak Usage')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot by model family
    if 'model_name' in successful.columns:
        successful['family'] = successful['model_name'].str.split('-').str[0]
        successful.boxplot(column='vram_peak_gb', by='family', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Model Family')
        axes[0, 1].set_ylabel('VRAM Peak (GB)')
        axes[0, 1].set_title('VRAM Usage by Model Family')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # VRAM vs Batch Size
    if 'batch_size' in successful.columns:
        axes[1, 0].scatter(successful['batch_size'], successful['vram_peak_gb'], 
                          alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('VRAM Peak (GB)')
        axes[1, 0].set_title('VRAM Usage vs Batch Size')
        axes[1, 0].grid(True, alpha=0.3)
    
    # VRAM vs Resolution
    if 'spatial_resolution' in successful.columns:
        axes[1, 1].scatter(successful['spatial_resolution'], successful['vram_peak_gb'],
                          alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Spatial Resolution')
        axes[1, 1].set_ylabel('VRAM Peak (GB)')
        axes[1, 1].set_title('VRAM Usage vs Spatial Resolution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_success_rate_by_category(df: pd.DataFrame, output_file: Optional[str] = None):
    """
    Plot success rate by different categories
    
    Args:
        df: DataFrame with results
        output_file: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rate by model
    if 'model_name' in df.columns:
        model_stats = df.groupby('model_name').agg({
            'success': ['sum', 'count']
        })
        model_stats.columns = ['successful', 'total']
        model_stats['success_rate'] = (model_stats['successful'] / model_stats['total'] * 100)
        model_stats = model_stats.sort_values('success_rate', ascending=True)
        
        axes[0, 0].barh(range(len(model_stats)), model_stats['success_rate'])
        axes[0, 0].set_yticks(range(len(model_stats)))
        axes[0, 0].set_yticklabels(model_stats.index, fontsize=8)
        axes[0, 0].set_xlabel('Success Rate (%)')
        axes[0, 0].set_title('Success Rate by Model')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Success rate by batch size
    if 'batch_size' in df.columns:
        batch_stats = df.groupby('batch_size').agg({
            'success': ['sum', 'count']
        })
        batch_stats.columns = ['successful', 'total']
        batch_stats['success_rate'] = (batch_stats['successful'] / batch_stats['total'] * 100)
        batch_stats = batch_stats.sort_index()
        
        axes[0, 1].plot(batch_stats.index, batch_stats['success_rate'], marker='o')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate by Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Success rate by resolution
    if 'spatial_resolution' in df.columns:
        res_stats = df.groupby('spatial_resolution').agg({
            'success': ['sum', 'count']
        })
        res_stats.columns = ['successful', 'total']
        res_stats['success_rate'] = (res_stats['successful'] / res_stats['total'] * 100)
        res_stats = res_stats.sort_index()
        
        axes[1, 0].plot(res_stats.index, res_stats['success_rate'], marker='o')
        axes[1, 0].set_xlabel('Spatial Resolution')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Success Rate by Spatial Resolution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Success rate by depth
    if 'depth' in df.columns:
        depth_stats = df.groupby('depth').agg({
            'success': ['sum', 'count']
        })
        depth_stats.columns = ['successful', 'total']
        depth_stats['success_rate'] = (depth_stats['successful'] / depth_stats['total'] * 100)
        depth_stats = depth_stats.sort_index()
        
        axes[1, 1].plot(depth_stats.index, depth_stats['success_rate'], marker='o')
        axes[1, 1].set_xlabel('Depth')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Success Rate by Depth')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_heatmap_vram_by_config(df: pd.DataFrame, output_file: Optional[str] = None):
    """
    Create heatmap of VRAM usage by batch size and resolution
    
    Args:
        df: DataFrame with successful results
        output_file: Optional path to save figure
    """
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0 or 'batch_size' not in successful.columns or 'spatial_resolution' not in successful.columns:
        print("Not enough data for heatmap")
        return
    
    # Create pivot table
    pivot = successful.pivot_table(
        values='vram_peak_gb',
        index='spatial_resolution',
        columns='batch_size',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'VRAM Peak (GB)'}, ax=ax)
    ax.set_title('Average VRAM Peak Usage by Batch Size and Spatial Resolution')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Spatial Resolution')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(df: pd.DataFrame, output_file: Optional[str] = None):
    """
    Compare models across different metrics
    
    Args:
        df: DataFrame with results
        output_file: Optional path to save figure
    """
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        print("No successful tests to plot")
        return
    
    if 'model_name' not in successful.columns:
        return
    
    model_stats = successful.groupby('model_name').agg({
        'vram_peak_gb': ['mean', 'max', 'min'],
        'duration_seconds': 'mean'
    }).round(2)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average VRAM by model
    model_stats[('vram_peak_gb', 'mean')].sort_values(ascending=True).plot(
        kind='barh', ax=axes[0], color='skyblue'
    )
    axes[0].set_xlabel('Average VRAM Peak (GB)')
    axes[0].set_title('Average VRAM Usage by Model')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Average duration by model
    model_stats[('duration_seconds', 'mean')].sort_values(ascending=True).plot(
        kind='barh', ax=axes[1], color='lightcoral'
    )
    axes[1].set_xlabel('Average Duration (seconds)')
    axes[1].set_title('Average Duration by Model')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_all_plots(df: pd.DataFrame, output_dir: str = "results/plots"):
    """
    Create all available plots and save to output directory
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    setup_style()
    
    print("Creating visualizations...")
    
    plot_vram_distribution(df, str(output_path / "vram_distribution.png"))
    plot_success_rate_by_category(df, str(output_path / "success_rates.png"))
    plot_heatmap_vram_by_config(df, str(output_path / "vram_heatmap.png"))
    plot_model_comparison(df, str(output_path / "model_comparison.png"))
    
    print(f"All plots saved to {output_dir}")
