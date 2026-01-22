"""Factory functions for loading models, optimizers, and dataloaders from checkpoints and configs."""

import sys
import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))
sys.path.insert(0, str(scripts_dir / "helpers"))

from scripts.core.model import create_model
from scripts.helpers.data_loader import create_dataloader


def load_model_from_checkpoint(checkpoint_path: Path, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load model from checkpoint, automatically recreating it with correct hyperparameters.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        device: Device to load model on (if None, uses checkpoint device or defaults to cpu)
        
    Returns:
        Model instance with weights loaded, ready to use
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If required hyperparameters are missing from checkpoint
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    
    # Extract hyperparameters
    in_channels = checkpoint.get('in_channels')
    num_classes = checkpoint.get('num_classes')
    
    if in_channels is None or num_classes is None:
        raise KeyError(
            f"Checkpoint missing required hyperparameters. "
            f"Found: in_channels={in_channels}, num_classes={num_classes}. "
            f"Please ensure checkpoint was saved with model hyperparameters."
        )
    
    # Create model with correct hyperparameters
    model = create_model(in_channels=in_channels, num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model


def load_optimizer_from_checkpoint(
    checkpoint_path: Path, 
    model: torch.nn.Module,
    device: Optional[torch.device] = None
) -> optim.Optimizer:
    """
    Load optimizer from checkpoint, automatically recreating it with correct hyperparameters.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        model: Model instance (optimizer will be created for this model's parameters)
        device: Device (for loading checkpoint, not used for optimizer)
        
    Returns:
        Optimizer instance with state loaded
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If required hyperparameters are missing from checkpoint
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    
    # Extract optimizer hyperparameters
    lr = checkpoint.get('learning_rate')
    weight_decay = checkpoint.get('weight_decay')
    
    if lr is None or weight_decay is None:
        raise KeyError(
            f"Checkpoint missing required optimizer hyperparameters. "
            f"Found: lr={lr}, weight_decay={weight_decay}. "
            f"Please ensure checkpoint was saved with optimizer hyperparameters."
        )
    
    # Create optimizer with correct hyperparameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return optimizer


def load_dataloader_from_config(
    config_path: Path,
    split: str = 'train',
    **override_kwargs
):
    """
    Load DataLoader from configuration file.
    
    Args:
        config_path: Path to config.yaml file
        split: 'train' or 'test'
        **override_kwargs: Override any config values (batch_size, num_workers, etc.)
        
    Returns:
        DataLoader instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    import yaml
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract data paths
    data_cfg = config.get('data', {})
    preprocessed_dir = data_cfg.get('preprocessed_dir')
    train_test_split_json = data_cfg.get('train_test_split_json')
    
    if not preprocessed_dir or not train_test_split_json:
        raise ValueError(
            f"Config missing required data paths. "
            f"Found: preprocessed_dir={preprocessed_dir}, train_test_split_json={train_test_split_json}"
        )
    
    # Build paths
    patches_dir = Path(preprocessed_dir) / "patches"
    patches_info_json = Path(preprocessed_dir) / "patches_info.json"
    
    # Extract training config for DataLoader parameters
    training_cfg = config.get('training', {})
    batch_size = override_kwargs.get('batch_size', training_cfg.get('batch_size', 64))
    num_workers = override_kwargs.get('num_workers', training_cfg.get('num_workers', 4))
    prefetch_factor = override_kwargs.get('prefetch_factor', training_cfg.get('prefetch_factor', 2))
    
    # Create DataLoader
    shuffle = (split == 'train')
    track_indices = (split == 'test')
    
    dataloader, _ = create_dataloader(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        track_indices=track_indices
    )
    
    return dataloader
