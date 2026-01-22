"""Configuration loading and argument parsing utilities."""

import yaml
import argparse
from pathlib import Path
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args(config: Dict) -> argparse.Namespace:
    """
    Parse command line arguments, using config values as defaults.
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        Parsed arguments namespace with CLI overrides applied
        
    Raises:
        ValueError: If required values are missing
    """
    parser = argparse.ArgumentParser(description='Train ResNet3D for binary classification')
    
    # Extract config sections
    training = config.get('training', {})
    model = config.get('model', {})
    input_cfg = config.get('input', {})
    system = config.get('system', {})
    data_cfg = config.get('data', {})
    
    # Run name (required)
    parser.add_argument('--run-name', type=str, required=True,
                       help='Name for this training run (will create _runs/{run_name}/)')
    
    # Data paths (from YAML only, CLI can override)
    default_preprocessed_dir = data_cfg.get('preprocessed_dir')
    default_train_test_split_json = data_cfg.get('train_test_split_json')
    parser.add_argument('--preprocessed-dir', type=str, default=default_preprocessed_dir,
                       help='Path to preprocessed directory (contains patches/ and patches_info.json)')
    parser.add_argument('--train-test-split-json', type=str, default=default_train_test_split_json,
                       help='Path to train_test_split.json file')
    
    # Training parameters (from YAML only, CLI can override)
    parser.add_argument('--batch-size', type=int, default=input_cfg.get('batch_size'),
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=training.get('epochs'),
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=training.get('learning_rate'),
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=training.get('weight_decay'),
                       help='Weight decay')
    parser.add_argument('--in-channels', type=int, default=model.get('in_channels'),
                       help='Number of input channels')
    
    parser.add_argument('--device', type=str, default=system.get('device'),
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=training.get('num_workers'),
                       help='Number of data loader workers')
    parser.add_argument('--prefetch-factor', type=int, default=training.get('prefetch_factor'),
                       help='Number of batches each worker prefetches in advance')
    
    parser.add_argument('--early-stopping-patience', type=int,
                       default=training.get('early_stopping_patience'),
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--early-stopping-min-delta', type=float,
                       default=training.get('early_stopping_min_delta'),
                       help='Minimum change to qualify as improvement')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., _runs/{run_name}/checkpoints/latest_model.pth)')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Validate that required values are present (from YAML or CLI)
    if args.preprocessed_dir is None:
        raise ValueError("Preprocessed directory must be specified in config.yaml (data.preprocessed_dir) or via --preprocessed-dir")
    if args.train_test_split_json is None:
        raise ValueError("Train/test split JSON must be specified in config.yaml (data.train_test_split_json) or via --train-test-split-json")
    
    return args
