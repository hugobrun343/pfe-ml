"""Analysis functions for training results."""

from collections import defaultdict
from typing import Dict, Tuple


def analyze_positions(results: dict) -> Dict[Tuple[int, int], dict]:
    """
    Analyze failure rate by grid position.
    
    Args:
        results: Training results dictionary
        
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


def analyze_combined(results: dict) -> Dict[Tuple[str, Tuple[int, int]], dict]:
    """
    Analyze failure rate by (volume, position) combination.
    
    Args:
        results: Training results dictionary
        
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
