"""
Data Loader - Load and filter results data
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def load_results(results_file: str) -> pd.DataFrame:
    """
    Load results from JSON or CSV file
    
    Args:
        results_file: Path to results file (CSV or JSON)
        
    Returns:
        DataFrame with results
    """
    results_path = Path(results_file)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    if results_path.suffix == '.json':
        with open(results_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif results_path.suffix == '.csv':
        df = pd.read_csv(results_path)
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")
    
    return df


def get_model_families() -> Dict[str, List[str]]:
    """
    Get mapping of model families to model names
    
    Returns:
        Dictionary mapping family names to lists of model names
    """
    return {
        'ResNet3D': ['ResNet3D-10', 'ResNet3D-18', 'ResNet3D-34', 'ResNet3D-50', 'ResNet3D-101'],
        'SE-ResNet3D': ['SE-ResNet3D-18', 'SE-ResNet3D-34', 'SE-ResNet3D-50', 'SE-ResNet3D-101', 'SE-ResNet3D-152'],
        'EfficientNet3D': ['EfficientNet3D-B0', 'EfficientNet3D-B1', 'EfficientNet3D-B2', 'EfficientNet3D-B3', 'EfficientNet3D-B4'],
        'ViT3D': ['ViT3D-Tiny', 'ViT3D-Small', 'ViT3D-Base', 'ViT3D-Large'],
        'ConvNeXt3D': ['ConvNeXt3D-Tiny', 'ConvNeXt3D-Small', 'ConvNeXt3D-Base', 'ConvNeXt3D-Large', 'ConvNeXt3D-XLarge'],
    }


def filter_by_model_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """
    Filter DataFrame by model family
    
    Args:
        df: DataFrame with results
        family: Model family name (e.g., 'ResNet3D', 'SE-ResNet3D')
        
    Returns:
        Filtered DataFrame
    """
    families = get_model_families()
    
    if family not in families:
        available = ', '.join(families.keys())
        raise ValueError(f"Unknown model family: {family}. Available: {available}")
    
    model_names = families[family]
    return df[df['model_name'].isin(model_names)].copy()


def filter_successful(df: pd.DataFrame) -> pd.DataFrame:
    """Filter only successful tests"""
    return df[df['success'] == True].copy()


def filter_failed(df: pd.DataFrame) -> pd.DataFrame:
    """Filter only failed tests"""
    return df[df['success'] == False].copy()


def filter_oom(df: pd.DataFrame) -> pd.DataFrame:
    """Filter OOM errors"""
    return df[df['error_message'] == 'OOM'].copy()


def get_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic statistics from DataFrame
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with statistics
    """
    total = len(df)
    successful = df['success'].sum() if 'success' in df.columns else 0
    failed = total - successful
    
    stats = {
        'total': total,
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / total * 100) if total > 0 else 0
    }
    
    if 'vram_peak_gb' in df.columns:
        successful_df = filter_successful(df)
        if len(successful_df) > 0:
            stats['vram'] = {
                'mean': successful_df['vram_peak_gb'].mean(),
                'median': successful_df['vram_peak_gb'].median(),
                'max': successful_df['vram_peak_gb'].max(),
                'min': successful_df['vram_peak_gb'].min(),
                'std': successful_df['vram_peak_gb'].std()
            }
    
    return stats
