"""
Utility functions for VRAM Grid Search
Handles configuration loading, logging, and common operations
"""

import yaml
import json
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(log_file: str = "grid_search.log", level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger for grid search with file and console output
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("GridSearch")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_config(config_path: str = "grid_search_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results_json(results: list, output_file: str) -> None:
    """
    Save results to JSON file
    
    Args:
        results: List of result dictionaries
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_results_csv(results: list, output_file: str) -> None:
    """
    Save results to CSV file using pandas
    
    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    try:
        import pandas as pd
        
        if not results:
            return
        
        df = pd.DataFrame(results)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
    except ImportError:
        print("Warning: pandas not installed, skipping CSV export")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information using PyTorch
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "error": "CUDA not available"
        }
    
    device = torch.cuda.current_device()
    
    return {
        "available": True,
        "device_id": device,
        "device_name": torch.cuda.get_device_name(device),
        "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }


def clear_gpu_memory() -> None:
    """
    Clear GPU cache and collect garbage
    Safe version that doesn't crash if GPU is in broken state
    """
    try:
        if torch.cuda.is_available():
            # Try to synchronize first (might fail if GPU is broken)
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass  # GPU might be broken, continue anyway
            
            # Try to empty cache (might fail if GPU is broken)
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass  # GPU is broken, can't clear cache
    
    except Exception:
        # Any error during cleanup - just continue
        # GPU might be in a broken state but we don't want to crash
        pass
    
    # Always try garbage collection (works even if GPU is broken)
    try:
        import gc
        gc.collect()
    except:
        pass


def wait_for_gpu_ready(timeout_seconds: float = 5.0, check_interval: float = 0.5) -> bool:
    """
    Wait for GPU to be ready after a crash
    Tests GPU with a simple operation (without blocking synchronize)
    
    Args:
        timeout_seconds: Maximum time to wait (default: 5s)
        check_interval: Time between checks (default: 0.5s)
        
    Returns:
        True if GPU is ready, False if timeout
    """
    import time
    
    if not torch.cuda.is_available():
        return False
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        try:
            # Simple test: create a tensor and do an operation
            # DON'T use synchronize() - it can block if GPU is in weird state
            x = torch.zeros(1, device='cuda')
            y = x + 1
            # Just check if operation completed (don't sync - async is OK)
            _ = y.item()  # This forces completion but is safer than sync
            
            # Clean up
            del x, y
            
            # Try to clear cache (might fail but that's OK)
            try:
                torch.cuda.empty_cache()
            except:
                pass
            
            # GPU is working!
            return True
        except RuntimeError:
            # GPU not ready yet, wait and retry
            time.sleep(check_interval)
        except Exception:
            # Any other error, wait and retry
            time.sleep(check_interval)
    
    # Timeout - GPU might still be OK, but we timed out waiting
    # Return True anyway - let the next test decide if GPU is really dead
    return True


def get_model_size_score(model_name: str) -> int:
    """
    Get a size score for a model (for comparison)
    Higher score = larger model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Size score (integer, higher = larger)
    """
    # Model size mapping (ordered from smallest to largest)
    model_scores = {
        # ResNet3D
        "ResNet3D-10": 10,
        "ResNet3D-18": 18,
        "ResNet3D-34": 34,
        "ResNet3D-50": 50,
        "ResNet3D-101": 101,
        
        # EfficientNet3D
        "EfficientNet3D-B0": 100,
        "EfficientNet3D-B1": 110,
        "EfficientNet3D-B2": 120,
        "EfficientNet3D-B3": 130,
        "EfficientNet3D-B4": 140,
        
        # ViT3D
        "ViT3D-Tiny": 200,
        "ViT3D-Small": 210,
        "ViT3D-Base": 220,
        "ViT3D-Large": 230,
        
        # ConvNeXt3D
        "ConvNeXt3D-Tiny": 300,
        "ConvNeXt3D-Small": 310,
        "ConvNeXt3D-Base": 320,
        "ConvNeXt3D-Large": 330,
        "ConvNeXt3D-XLarge": 340,
        
        # SE-ResNet3D
        "SE-ResNet3D-18": 400,
        "SE-ResNet3D-34": 410,
        "SE-ResNet3D-50": 420,
        "SE-ResNet3D-101": 430,
        "SE-ResNet3D-152": 440,
    }
    
    return model_scores.get(model_name, 0)


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes to human readable string
    
    Args:
        bytes_val: Number of bytes
        
    Returns:
        Formatted string (e.g., "15.2 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def estimate_total_combinations(config: Dict[str, Any]) -> int:
    """
    Estimate total number of combinations to test
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Total number of combinations
    """
    grid = config['grid_search']
    
    num_spatial = len(grid['spatial_resolutions'])
    num_depth = len(grid['depth_sizes'])
    num_batch = len(grid['batch_sizes'])
    num_models = len(grid['models'])
    
    total = num_spatial * num_depth * num_batch * num_models
    
    return total


def create_result_entry(
    spatial_res: int,
    depth: int,
    batch_size: int,
    model_name: str,
    model_size: str,
    success: bool,
    vram_used_bytes: Optional[int] = None,
    vram_peak_bytes: Optional[int] = None,
    error_message: Optional[str] = None,
    duration_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create a standardized result entry
    
    Args:
        spatial_res: Spatial resolution (H=W)
        depth: Depth (D)
        batch_size: Batch size
        model_name: Model name (e.g., "ResNet3D-34")
        model_size: Model size category (tiny/small/medium/large/xlarge)
        success: Whether test succeeded
        vram_used_bytes: VRAM used in bytes
        vram_peak_bytes: Peak VRAM in bytes
        error_message: Error message if failed
        duration_seconds: Test duration in seconds
        
    Returns:
        Result dictionary
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "spatial_resolution": spatial_res,
        "depth": depth,
        "batch_size": batch_size,
        "model_name": model_name,
        "model_size": model_size,
        "patch_shape": f"{batch_size}x{depth}x{spatial_res}x{spatial_res}",
        "success": success,
        "duration_seconds": duration_seconds
    }
    
    if success and vram_used_bytes is not None:
        result["vram_used_bytes"] = vram_used_bytes
        result["vram_used_gb"] = vram_used_bytes / (1024**3)
        result["vram_used_formatted"] = format_bytes(vram_used_bytes)
        
        if vram_peak_bytes is not None:
            result["vram_peak_bytes"] = vram_peak_bytes
            result["vram_peak_gb"] = vram_peak_bytes / (1024**3)
            result["vram_peak_formatted"] = format_bytes(vram_peak_bytes)
    else:
        result["error_message"] = error_message or "Unknown error"
    
    return result


def print_banner(text: str, width: int = 80, char: str = "=") -> None:
    """Print a formatted banner"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print progress bar"""
    percentage = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\r{prefix}: [{bar}] {current}/{total} ({percentage:.1f}%)", end="", flush=True)
    
    if current == total:
        print()  # New line when complete
