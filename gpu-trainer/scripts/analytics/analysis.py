"""Analysis functions for training results."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

from scripts.analytics.io import load_preprocessing_metadata, calculate_grid_position


def analyze_positions(
    results: dict,
    preprocessing_metadata: Optional[Dict] = None
) -> Dict[Tuple[int, int], dict]:
    """
    Analyze failure rate by grid position.
    
    For mode 'max': calculates grid positions (i, j) from position_h/w
    For mode 'top_n': uses patch_index for grouping (no grid visualization)
    
    Args:
        results: Training results dictionary
        preprocessing_metadata: Optional preprocessing metadata to get patch dimensions
        
    Returns:
        Dict mapping (i, j) -> {'total': int, 'failures': int, 'failure_rate': float}
        For mode 'top_n', returns empty dict (use analyze_patch_indices instead)
    """
    position_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    # Use the last epoch for analysis (best model)
    last_epoch = results['epochs'][-1]
    if 'validation' not in last_epoch or 'samples' not in last_epoch['validation']:
        return {}
    
    samples = last_epoch['validation']['samples']
    
    # Get preprocessing config if available
    patch_mode = None
    target_height = None
    target_width = None
    
    if preprocessing_metadata:
        config = preprocessing_metadata.get('config', {})
        patch_extraction = config.get('patch_extraction', {})
        patch_mode = patch_extraction.get('mode', 'max')
        target_height = config.get('target_height')
        target_width = config.get('target_width')
    
    # If mode is 'top_n', we can't create a grid visualization
    if patch_mode == 'top_n':
        return {}
    
    # For mode 'max', calculate grid positions from position_h/w
    for sample in samples:
        if 'position_h' not in sample or 'position_w' not in sample:
            continue
        
        if target_height and target_width:
            i, j = calculate_grid_position(
                sample['position_h'],
                sample['position_w'],
                target_height,
                target_width
            )
        else:
            # Fallback: can't calculate without metadata
            continue
        
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
    Analyze failure rate by volume (stack_id).
    
    Args:
        results: Training results dictionary
        
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


def analyze_combined(
    results: dict,
    preprocessing_metadata: Optional[Dict] = None
) -> Dict[Tuple[str, Tuple[int, int]], dict]:
    """
    Analyze failure rate by (volume, position) combination.
    
    Args:
        results: Training results dictionary
        preprocessing_metadata: Optional preprocessing metadata to get patch dimensions
        
    Returns:
        Dict mapping (stack_id, (i, j)) -> {'total': int, 'failures': int, 'failure_rate': float}
    """
    combined_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    # Use the last epoch for analysis
    last_epoch = results['epochs'][-1]
    if 'validation' not in last_epoch or 'samples' not in last_epoch['validation']:
        return {}
    
    samples = last_epoch['validation']['samples']
    
    # Get preprocessing config if available
    patch_mode = None
    target_height = None
    target_width = None
    
    if preprocessing_metadata:
        config = preprocessing_metadata.get('config', {})
        patch_extraction = config.get('patch_extraction', {})
        patch_mode = patch_extraction.get('mode', 'max')
        target_height = config.get('target_height')
        target_width = config.get('target_width')
    
    # If mode is 'top_n', we can't create a grid visualization
    if patch_mode == 'top_n':
        return {}
    
    for sample in samples:
        if 'stack_id' not in sample or 'position_h' not in sample or 'position_w' not in sample:
            continue
        
        stack_id = sample['stack_id']
        
        if target_height and target_width:
            i, j = calculate_grid_position(
                sample['position_h'],
                sample['position_w'],
                target_height,
                target_width
            )
        else:
            # Fallback: can't calculate without metadata
            continue
        
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


def analyze_patch_indices(results: dict) -> Dict[int, dict]:
    """
    Analyze failure rate by patch index (for mode 'top_n').
    
    Args:
        results: Training results dictionary
        
    Returns:
        Dict mapping patch_index -> {'total': int, 'failures': int, 'failure_rate': float}
    """
    index_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    # Use the last epoch for analysis
    last_epoch = results['epochs'][-1]
    if 'validation' not in last_epoch or 'samples' not in last_epoch['validation']:
        return {}
    
    samples = last_epoch['validation']['samples']
    
    for sample in samples:
        if 'patch_index' not in sample:
            continue
        
        patch_idx = sample['patch_index']
        index_stats[patch_idx]['total'] += 1
        if not sample['correct']:
            index_stats[patch_idx]['failures'] += 1
    
    # Calculate failure rates
    index_analysis = {}
    for idx, stats in index_stats.items():
        index_analysis[idx] = {
            'total': stats['total'],
            'failures': stats['failures'],
            'failure_rate': stats['failures'] / stats['total'] if stats['total'] > 0 else 0.0
        }
    
    return index_analysis
