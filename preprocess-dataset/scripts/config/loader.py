"""Configuration loading and resolution."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate preprocessing configuration from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "version" not in config:
        raise ValueError("'version' field is required in config file")
    if "paths" not in config:
        raise ValueError("'paths' section is required in config file")
    if "preprocessing" not in config:
        raise ValueError("'preprocessing' section is required in config file")
    return config


def resolve_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Path]:
    """Resolve all paths in configuration."""
    paths = config["paths"]
    resolved = {}
    dataset_json_path = Path(paths["dataset_json"])
    resolved["dataset_json"] = dataset_json_path if dataset_json_path.is_absolute() else (config_path.parent.parent / paths["dataset_json"]).resolve()
    resolved["data_root"] = Path(paths["data_root"])
    output_dir_path = Path(paths["output_dir"])
    resolved["output_dir"] = output_dir_path if output_dir_path.is_absolute() else (config_path.parent.parent / paths["output_dir"]).resolve()
    return resolved


def get_slice_selection_method(cfg: Dict[str, Any]) -> Tuple[str, Optional[Union[float, list]], Optional[Union[float, list]]]:
    """Extract slice selection method and parameters from config."""
    slice_selection = cfg.get("slice_selection", "intensity")
    if isinstance(slice_selection, dict):
        return (
            slice_selection.get("method", "intensity"),
            slice_selection.get("min_intensity"),
            slice_selection.get("max_intensity"),
        )
    return slice_selection, None, None


def get_patch_extraction_config(cfg: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[int]]:
    """Extract patch extraction mode and parameters from config."""
    if "patch_extraction" not in cfg:
        raise ValueError("'patch_extraction' section is required")
    pc = cfg["patch_extraction"]
    mode = pc.get("mode", "max")
    if mode == "max":
        return "max", None, None
    if mode == "top_n":
        n_patches = pc.get("n_patches")
        pool_stride = pc.get("pool_stride", 2)
        if n_patches is None:
            raise ValueError("'n_patches' is required when mode == 'top_n'")
        if pool_stride not in (1, 2, 3):
            raise ValueError(f"pool_stride must be 1, 2, or 3, got {pool_stride}")
        return "top_n", n_patches, pool_stride
    raise ValueError(f"Unknown patch_extraction.mode: {mode}")


def get_normalization_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalization configuration from main config."""
    norm_cfg = cfg.get("normalization", {})
    return {
        "method": norm_cfg.get("method", "z-score"),
        "clip_min": norm_cfg.get("clip_min"),
        "clip_max": norm_cfg.get("clip_max"),
        "scale_below_range": norm_cfg.get("scale_below_range"),
        "scale_middle_range": norm_cfg.get("scale_middle_range"),
        "scale_above_range": norm_cfg.get("scale_above_range"),
        "percentile_low": norm_cfg.get("percentile_low", 1),
        "percentile_high": norm_cfg.get("percentile_high", 99),
    }


def load_context(config_path: Path) -> Dict[str, Any]:
    """Load config, resolve paths, and return full context for the pipeline."""
    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    cfg = config["preprocessing"]
    patch_mode, n_patches, pool_stride = get_patch_extraction_config(cfg)
    return {
        "config": config,
        "paths": paths,
        "cfg": cfg,
        "version": config["version"],
        "norm_config": get_normalization_config(cfg),
        "patch_mode": patch_mode,
        "n_patches": n_patches,
        "pool_stride": pool_stride,
    }


def get_output_dirs(context: Dict[str, Any]) -> Tuple[Path, Path]:
    """Create and return output_base and patches_output from context."""
    cfg = context["cfg"]
    paths = context["paths"]
    version = context["version"]
    h, w, d = cfg["target_height"], cfg["target_width"], cfg["target_depth"]
    output_base = paths["output_dir"] / f"preprocessed_{h}x{w}x{d}_{version}"
    patches_output = output_base / "patches"
    patches_output.mkdir(parents=True, exist_ok=True)
    return output_base, patches_output
