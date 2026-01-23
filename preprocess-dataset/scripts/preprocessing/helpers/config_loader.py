"""Configuration loading and validation utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate preprocessing configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if 'version' not in config:
        raise ValueError("'version' field is required in config file (e.g., 'v0', 'v1', 'v2')")
    
    if 'paths' not in config:
        raise ValueError("'paths' section is required in config file")
    
    if 'preprocessing' not in config:
        raise ValueError("'preprocessing' section is required in config file")
    
    return config


def resolve_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Path]:
    """
    Resolve all paths in configuration (handle absolute and relative paths).
    
    Args:
        config: Configuration dictionary
        config_path: Path to the config file (for resolving relative paths)
        
    Returns:
        Dictionary with resolved Path objects
    """
    paths = config['paths']
    resolved = {}
    
    # Dataset JSON path
    dataset_json_path = Path(paths['dataset_json'])
    if dataset_json_path.is_absolute():
        resolved['dataset_json'] = dataset_json_path
    else:
        resolved['dataset_json'] = (config_path.parent.parent / paths['dataset_json']).resolve()
    
    # Data root path
    resolved['data_root'] = Path(paths['data_root'])
    
    # Output directory path
    output_dir_path = Path(paths['output_dir'])
    if output_dir_path.is_absolute():
        resolved['output_dir'] = output_dir_path
    else:
        resolved['output_dir'] = (config_path.parent.parent / paths['output_dir']).resolve()
    
    return resolved


def get_slice_selection_method(cfg: Dict[str, Any]) -> tuple[str, Optional[float], Optional[float]]:
    """
    Extract slice selection method and parameters from config.
    
    Args:
        cfg: Preprocessing configuration dictionary
        
    Returns:
        Tuple of (method, min_intensity, max_intensity)
    """
    slice_selection = cfg.get('slice_selection', 'intensity')
    if isinstance(slice_selection, dict):
        method = slice_selection.get('method', 'intensity')
        min_intensity = slice_selection.get('min_intensity')
        max_intensity = slice_selection.get('max_intensity')
        return method, min_intensity, max_intensity
    return slice_selection, None, None


def get_patch_extraction_config(cfg: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Extract patch extraction mode and parameters from config.
    
    Args:
        cfg: Preprocessing configuration dictionary
        
    Returns:
        Tuple of (mode, n_patches, scoring_method)
        - mode: 'max' or 'top_n'
        - n_patches: Number of patches (None for 'max' mode)
        - scoring_method: Scoring method for 'top_n' mode (None for 'max' mode)
        
    Raises:
        ValueError: If 'patch_extraction' section is missing or invalid
    """
    if 'patch_extraction' not in cfg:
        raise ValueError("'patch_extraction' section is required in preprocessing configuration")
    
    patch_config = cfg['patch_extraction']
    mode = patch_config.get('mode', 'max')
    
    if mode == 'max':
        return 'max', None, None
    elif mode == 'top_n':
        n_patches = patch_config.get('n_patches')
        scoring_method = patch_config.get('scoring_method', 'intensity')
        if n_patches is None:
            raise ValueError("'n_patches' is required when patch_extraction.mode == 'top_n'")
        return 'top_n', n_patches, scoring_method
    else:
        raise ValueError(f"Unknown patch_extraction.mode: {mode}. Must be 'max' or 'top_n'")
