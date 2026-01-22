"""
Best Configurations Extractor - Find best performing configurations
"""

import pandas as pd
from typing import Dict, Optional


def get_top_by_vram(df: pd.DataFrame, n: int = 50, min_vram: Optional[float] = None) -> pd.DataFrame:
    """
    Get top N configurations by VRAM peak (highest VRAM that fits)
    
    Args:
        df: DataFrame with successful results
        n: Number of top configurations to return
        min_vram: Minimum VRAM threshold (optional)
        
    Returns:
        DataFrame with top N configurations
    """
    successful = df[df['success'] == True].copy()
    
    if min_vram is not None:
        successful = successful[successful['vram_peak_gb'] >= min_vram]
    
    if len(successful) == 0:
        return pd.DataFrame()
    
    top = successful.nlargest(n, 'vram_peak_gb')
    return top[['model_name', 'batch_size', 'spatial_resolution', 'depth', 
                'patch_shape', 'vram_peak_gb', 'vram_used_gb', 'duration_seconds']]


def get_best_by_efficiency(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """
    Get best configurations by efficiency (high VRAM usage, low duration)
    
    Args:
        df: DataFrame with successful results
        n: Number of configurations to return
        
    Returns:
        DataFrame with best efficient configurations
    """
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        return pd.DataFrame()
    
    # Normalize metrics (0-1 scale)
    vram_norm = (successful['vram_peak_gb'] - successful['vram_peak_gb'].min()) / \
                (successful['vram_peak_gb'].max() - successful['vram_peak_gb'].min() + 1e-10)
    duration_norm = 1 - (successful['duration_seconds'] - successful['duration_seconds'].min()) / \
                    (successful['duration_seconds'].max() - successful['duration_seconds'].min() + 1e-10)
    
    # Efficiency score: weighted combination
    successful['efficiency_score'] = 0.7 * vram_norm + 0.3 * duration_norm
    
    best = successful.nlargest(n, 'efficiency_score')
    return best[['model_name', 'batch_size', 'spatial_resolution', 'depth',
                 'patch_shape', 'vram_peak_gb', 'duration_seconds', 'efficiency_score']]


def get_best_by_batch_size(df: pd.DataFrame, batch_size: int, n: int = 10) -> pd.DataFrame:
    """
    Get best configurations for a specific batch size
    
    Args:
        df: DataFrame with successful results
        batch_size: Target batch size
        n: Number of configurations to return
        
    Returns:
        DataFrame with best configurations for this batch size
    """
    successful = df[df['success'] == True].copy()
    batch_df = successful[successful['batch_size'] == batch_size].copy()
    
    if len(batch_df) == 0:
        return pd.DataFrame()
    
    top = batch_df.nlargest(n, 'vram_peak_gb')
    return top[['model_name', 'batch_size', 'spatial_resolution', 'depth',
                'patch_shape', 'vram_peak_gb', 'duration_seconds']]


def get_best_by_resolution(df: pd.DataFrame, resolution: int, n: int = 10) -> pd.DataFrame:
    """
    Get best configurations for a specific spatial resolution
    
    Args:
        df: DataFrame with successful results
        resolution: Target spatial resolution
        n: Number of configurations to return
        
    Returns:
        DataFrame with best configurations for this resolution
    """
    successful = df[df['success'] == True].copy()
    res_df = successful[successful['spatial_resolution'] == resolution].copy()
    
    if len(res_df) == 0:
        return pd.DataFrame()
    
    top = res_df.nlargest(n, 'vram_peak_gb')
    return top[['model_name', 'batch_size', 'spatial_resolution', 'depth',
                'patch_shape', 'vram_peak_gb', 'duration_seconds']]


def get_failure_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze failure patterns
    
    Args:
        df: DataFrame with all results
        
    Returns:
        Dictionary with failure analysis
    """
    failed = df[df['success'] == False].copy()
    
    if len(failed) == 0:
        return {'total_failures': 0}
    
    analysis = {
        'total_failures': len(failed),
        'oom_count': len(failed[failed['error_message'] == 'OOM']) if 'error_message' in failed.columns else 0,
        'skipped_count': len(failed[failed['error_message'].str.contains('SKIPPED', na=False)]) if 'error_message' in failed.columns else 0,
    }
    
    # Failures by model
    if 'model_name' in failed.columns:
        analysis['failures_by_model'] = failed['model_name'].value_counts().to_dict()
    
    # Failures by batch size
    if 'batch_size' in failed.columns:
        analysis['failures_by_batch'] = failed['batch_size'].value_counts().to_dict()
    
    # Failures by resolution
    if 'spatial_resolution' in failed.columns:
        analysis['failures_by_resolution'] = failed['spatial_resolution'].value_counts().to_dict()
    
    # Most common failing configurations
    if all(col in failed.columns for col in ['model_name', 'batch_size', 'spatial_resolution', 'depth']):
        config_cols = ['model_name', 'batch_size', 'spatial_resolution', 'depth']
        grouped = failed.groupby(config_cols).size().sort_values(ascending=False).head(10)
        # Convert tuple keys to strings for JSON serialization
        analysis['common_failing_configs'] = {str(k): int(v) for k, v in grouped.items()}
    
    return analysis


def get_all_best_combinations(df: pd.DataFrame, n_per_category: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Get best combinations across multiple categories
    
    Args:
        df: DataFrame with results
        n_per_category: Number of best configs per category
        
    Returns:
        Dictionary with different categories of best configs
    """
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        return {}
    
    results = {
        'top_by_vram': get_top_by_vram(df, n=n_per_category),
        'top_by_efficiency': get_best_by_efficiency(df, n=n_per_category),
    }
    
    # Best by model
    if 'model_name' in successful.columns:
        by_model = []
        for model in successful['model_name'].unique():
            model_df = successful[successful['model_name'] == model]
            top_model = model_df.nlargest(5, 'vram_peak_gb')
            by_model.append(top_model)
        if by_model:
            results['top_by_model'] = pd.concat(by_model, ignore_index=True)
    
    # Best by batch size
    if 'batch_size' in successful.columns:
        by_batch = []
        for batch in sorted(successful['batch_size'].unique()):
            batch_df = successful[successful['batch_size'] == batch]
            top_batch = batch_df.nlargest(5, 'vram_peak_gb')
            by_batch.append(top_batch)
        if by_batch:
            results['top_by_batch'] = pd.concat(by_batch, ignore_index=True)
    
    return results
