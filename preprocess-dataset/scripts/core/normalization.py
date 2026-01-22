"""Modular normalization functions for patch preprocessing."""

import numpy as np
from typing import Dict, Any, Optional, Tuple


def compute_stats_on_sample(patches: list, method: str = 'percentile', normalize_globally: bool = False) -> Dict[str, float]:
    """
    Compute normalization statistics on a representative sample of patches.
    
    Args:
        patches: List of patch arrays
        method: 'percentile', 'minmax', 'zscore', 'robust'
        normalize_globally: If True, compute global stats for all patches (for global normalization)
        
    Returns:
        Dictionary with computed statistics
    """
    # Flatten all patches and compute stats
    if normalize_globally:
        # Use all patches for global normalization
        all_values = np.concatenate([p.flatten() for p in patches])
    else:
        # Use sample for percentile/robust methods
        all_values = np.concatenate([p.flatten() for p in patches[:min(100, len(patches))]])
    
    stats = {
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'median': float(np.median(all_values))
    }
    
    if normalize_globally:
        # Add global prefix for global normalization
        stats['global_min'] = stats['min']
        stats['global_max'] = stats['max']
        stats['global_mean'] = stats['mean']
        stats['global_std'] = stats['std']
        stats['global_median'] = stats['median']
        
        # Compute IQR for robust normalization
        q75 = np.percentile(all_values, 75)
        q25 = np.percentile(all_values, 25)
        stats['global_iqr'] = float(q75 - q25)
    
    if method == 'percentile' or normalize_globally:
        stats['percentile_1'] = float(np.percentile(all_values, 1))
        stats['percentile_99'] = float(np.percentile(all_values, 99))
        stats['percentile_5'] = float(np.percentile(all_values, 5))
        stats['percentile_95'] = float(np.percentile(all_values, 95))
        
        if normalize_globally:
            stats['global_percentile_1'] = stats['percentile_1']
            stats['global_percentile_99'] = stats['percentile_99']
    
    return stats


def normalize_patch(
    patch: np.ndarray,
    method: str = 'z-score',
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    below_clip_value: Optional[float] = None,
    above_clip_value: Optional[float] = None,
    scale_below_range: Optional[Tuple[float, float]] = None,
    scale_above_range: Optional[Tuple[float, float]] = None,
    scale_middle_range: Optional[Tuple[float, float]] = None,
    stats: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Normalize a patch with various methods.
    
    Args:
        patch: Patch array (H, W, D, C)
        method: 'z-score', 'min-max', 'robust', 'percentile'
        clip_min: Main clipping minimum (plage principale à garder)
        clip_max: Main clipping maximum (plage principale à garder)
        below_clip_value: Valeur fixe pour les valeurs < clip_min (si None, utilise scale_below_range)
        above_clip_value: Valeur fixe pour les valeurs > clip_max (si None, utilise scale_above_range)
        scale_below_range: Plage cible [min, max] pour redimensionner [min_patch, clip_min] (ex: [0, 0.1])
        scale_above_range: Plage cible [min, max] pour redimensionner [clip_max, max_patch] (ex: [0.9, 1])
        scale_middle_range: Plage cible [min, max] pour redimensionner [clip_min, clip_max] (ex: [0.1, 0.9])
        stats: Optional precomputed statistics (for global normalization or percentile/robust methods)
        
    Returns:
        Normalized patch
    """
    patch = patch.astype(np.float32)
    patch_min = float(np.min(patch))
    patch_max = float(np.max(patch))
    
    # Apply clipping/remapping if specified
    if clip_min is not None or clip_max is not None:
        result = np.zeros_like(patch)
        
        # Déterminer les plages
        actual_clip_min = clip_min if clip_min is not None else patch_min
        actual_clip_max = clip_max if clip_max is not None else patch_max
        
        # Masques pour chaque région
        mask_below = patch < actual_clip_min
        mask_middle = (patch >= actual_clip_min) & (patch <= actual_clip_max)
        mask_above = patch > actual_clip_max
        
        # Traiter les valeurs < clip_min
        if np.any(mask_below):
            if below_clip_value is not None:
                # Valeur fixe
                result[mask_below] = below_clip_value
            elif scale_below_range is not None:
                # Redimensionner [patch_min, clip_min] dans [scale_below_range[0], scale_below_range[1]]
                below_min, below_max = scale_below_range
                if actual_clip_min > patch_min:
                    result[mask_below] = below_min + (patch[mask_below] - patch_min) * (below_max - below_min) / (actual_clip_min - patch_min)
                else:
                    result[mask_below] = below_min
            else:
                # Par défaut: mettre à clip_min
                result[mask_below] = actual_clip_min
        
        # Traiter les valeurs entre clip_min et clip_max
        if np.any(mask_middle):
            if scale_middle_range is not None:
                # Redimensionner [clip_min, clip_max] dans scale_middle_range
                middle_min, middle_max = scale_middle_range
                if actual_clip_max > actual_clip_min:
                    result[mask_middle] = middle_min + (patch[mask_middle] - actual_clip_min) * (middle_max - middle_min) / (actual_clip_max - actual_clip_min)
                else:
                    result[mask_middle] = middle_min
            else:
                # Garder les valeurs telles quelles
                result[mask_middle] = patch[mask_middle]
        
        # Traiter les valeurs > clip_max
        if np.any(mask_above):
            if above_clip_value is not None:
                # Valeur fixe
                result[mask_above] = above_clip_value
            elif scale_above_range is not None:
                # Redimensionner [clip_max, patch_max] dans [scale_above_range[0], scale_above_range[1]]
                above_min, above_max = scale_above_range
                if patch_max > actual_clip_max:
                    result[mask_above] = above_min + (patch[mask_above] - actual_clip_max) * (above_max - above_min) / (patch_max - actual_clip_max)
                else:
                    result[mask_above] = above_min
            else:
                # Par défaut: mettre à clip_max
                result[mask_above] = actual_clip_max
        
        patch = result
    
    # Apply normalization using global stats if provided, otherwise patch-local stats
    if method == 'z-score':
        if stats and 'global_mean' in stats and 'global_std' in stats:
            # Normalisation globale
            mean = stats['global_mean']
            std = stats['global_std']
        else:
            # Normalisation locale (patch par patch)
            mean = np.mean(patch)
            std = np.std(patch)
        
        if std > 0:
            patch = (patch - mean) / std
        else:
            patch = patch - mean
            
    elif method == 'min-max':
        if stats and 'global_min' in stats and 'global_max' in stats:
            # Normalisation globale
            vmin = stats['global_min']
            vmax = stats['global_max']
        else:
            # Normalisation locale (patch par patch)
            vmin = np.min(patch)
            vmax = np.max(patch)
        
        if vmax > vmin:
            patch = (patch - vmin) / (vmax - vmin)
        else:
            patch = np.zeros_like(patch)
            
    elif method == 'robust':
        # Use median and IQR (Interquartile Range)
        if stats and 'global_median' in stats and 'global_iqr' in stats:
            # Normalisation globale
            median = stats['global_median']
            iqr = stats['global_iqr']
        else:
            # Normalisation locale
            median = np.median(patch)
            q75 = np.percentile(patch, 75)
            q25 = np.percentile(patch, 25)
            iqr = q75 - q25
        
        if iqr > 0:
            patch = (patch - median) / iqr
        else:
            patch = patch - median
            
    elif method == 'percentile':
        # Normalize using percentile-based range
        if stats:
            p_low = stats.get('percentile_1', stats.get('global_percentile_1', np.percentile(patch, 1)))
            p_high = stats.get('percentile_99', stats.get('global_percentile_99', np.percentile(patch, 99)))
        else:
            p_low = np.percentile(patch, 1)
            p_high = np.percentile(patch, 99)
        
        if p_high > p_low:
            patch = np.clip(patch, p_low, p_high)
            patch = (patch - p_low) / (p_high - p_low)
        else:
            patch = patch - p_low
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return patch


def get_normalization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract normalization configuration from main config.
    
    Args:
        config: Main preprocessing config dictionary
        
    Returns:
        Normalization config dictionary
    """
    norm_config = config.get('normalization', {})
    
    # Support both old format (string) and new format (dict)
    if isinstance(norm_config, str):
        return {
            'method': norm_config,
            'clip_min': None,
            'clip_max': None,
            'below_clip_value': None,
            'above_clip_value': None,
            'scale_below_range': None,
            'scale_above_range': None,
            'scale_middle_range': None,
            'normalize_globally': False,
            'percentile_low': 1,
            'percentile_high': 99,
            'compute_on_sample': False
        }
    
    # Parse ranges if provided as lists
    def parse_range(r):
        if r is not None and isinstance(r, list):
            return tuple(r)
        return r
    
    return {
        'method': norm_config.get('method', 'z-score'),
        'clip_min': norm_config.get('clip_min'),
        'clip_max': norm_config.get('clip_max'),
        'below_clip_value': norm_config.get('below_clip_value'),
        'above_clip_value': norm_config.get('above_clip_value'),
        'scale_below_range': parse_range(norm_config.get('scale_below_range')),
        'scale_above_range': parse_range(norm_config.get('scale_above_range')),
        'scale_middle_range': parse_range(norm_config.get('scale_middle_range')),
        'normalize_globally': norm_config.get('normalize_globally', False),
        'percentile_low': norm_config.get('percentile_low', 1),
        'percentile_high': norm_config.get('percentile_high', 99),
        'compute_on_sample': norm_config.get('compute_on_sample', False)
    }
