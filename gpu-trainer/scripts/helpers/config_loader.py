"""Configuration loading and argument parsing utilities."""

import sys
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
    Parse command line arguments. Only --config and --resume are CLI arguments.
    All other parameters come strictly from config.yaml.
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        Parsed arguments namespace with all values from config
        
    Raises:
        ValueError: If required values are missing in config
    """
    class _Parser(argparse.ArgumentParser):
        def error(self, message):
            self.print_usage(sys.stderr)
            self.exit(2, f"Error: {message}\n"
                         "Only --config and --resume are accepted on the command line. "
                         "All other parameters (run_name, batch_size, etc.) must be set in the config file.\n")

    parser = _Parser(
        description='Train ResNet3D for binary classification.',
        epilog='Only --config and --resume are accepted on the command line. '
               'All other parameters (run_name, batch_size, num_workers, etc.) must be set in the config file.',
    )
    
    # Only CLI arguments accepted: config and resume
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file (required)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Chemin vers un checkpoint pour reprendre l’entraînement (optional)')
    
    # Parse only CLI args (config and resume)
    cli_args = parser.parse_args()
    
    # Support flat format (key: { value: x }) when sections are missing
    def _flat_val(cfg: Dict, key: str):
        v = cfg.get(key)
        if isinstance(v, dict) and 'value' in v:
            return v['value']
        return v

    if not config.get('training') and isinstance(config.get('batch_size'), dict) and 'value' in config.get('batch_size', {}):
        training = {
            'epochs': _flat_val(config, 'epochs'),
            'learning_rate': _flat_val(config, 'learning_rate'),
            'weight_decay': _flat_val(config, 'weight_decay'),
            'num_workers': _flat_val(config, 'num_workers'),
            'prefetch_factor': _flat_val(config, 'prefetch_factor'),
            'early_stopping_patience': _flat_val(config, 'early_stopping_patience'),
            'early_stopping_min_delta': _flat_val(config, 'early_stopping_min_delta'),
        }
        model = {'in_channels': _flat_val(config, 'in_channels')}
        input_cfg = {'batch_size': _flat_val(config, 'batch_size')}
        system = {'device': _flat_val(config, 'device')}
        data_cfg = {
            'preprocessed_dir': _flat_val(config, 'preprocessed_dir'),
            'train_test_split_json': _flat_val(config, 'train_test_split_json'),
        }
        run_cfg = {'run_name': _flat_val(config, 'run_name') or 'resnet3d'}
    else:
        training = config.get('training', {})
        model = config.get('model', {})
        input_cfg = config.get('input', {})
        system = config.get('system', {})
        data_cfg = config.get('data', {})
        run_cfg = config.get('run', {})
    
    # Build args namespace from config only (no CLI override)
    args = argparse.Namespace()
    
    # CLI-only args
    args.config = cli_args.config
    args.resume = cli_args.resume
    
    # All other args come strictly from config
    args.run_name = run_cfg.get('run_name')
    args.group = run_cfg.get('group')
    args.preprocessed_dir = data_cfg.get('preprocessed_dir')
    args.train_test_split_json = data_cfg.get('train_test_split_json')
    args.batch_size = input_cfg.get('batch_size')
    args.epochs = training.get('epochs')
    args.lr = training.get('learning_rate')
    args.weight_decay = training.get('weight_decay')
    args.model_name = model.get('name', 'resnet3d_50')  # Default to resnet3d_50
    args.in_channels = model.get('in_channels', 3)
    args.device = system.get('device')
    args.num_workers = training.get('num_workers')
    args.prefetch_factor = training.get('prefetch_factor')
    args.early_stopping_patience = training.get('early_stopping_patience')
    args.early_stopping_min_delta = training.get('early_stopping_min_delta')
    
    # Validate that required values are present in config
    if args.run_name is None:
        raise ValueError("Run name must be specified in config.yaml (run.run_name)")
    if args.preprocessed_dir is None:
        raise ValueError("Preprocessed directory must be specified in config.yaml (data.preprocessed_dir)")
    if args.train_test_split_json is None:
        raise ValueError("Train/test split JSON must be specified in config.yaml (data.train_test_split_json)")
    if args.batch_size is None:
        raise ValueError("Batch size must be specified in config.yaml (input.batch_size)")
    if args.epochs is None:
        raise ValueError("Epochs must be specified in config.yaml (training.epochs)")
    if args.lr is None:
        raise ValueError("Learning rate must be specified in config.yaml (training.learning_rate)")
    if args.num_workers is None:
        raise ValueError("num_workers must be specified in config.yaml (training.num_workers)")
    if args.device is None:
        raise ValueError("Device must be specified in config.yaml (system.device)")
    
    return args
