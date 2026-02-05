"""Data loading utilities for training."""

import sys
from pathlib import Path
from argparse import Namespace
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "helpers"))

from scripts.helpers.data_loader import create_dataloader


def load_train_val_data(args: Namespace):
    """
    Load training and validation data loaders from .nii.gz or .npy patches.
    
    Args:
        args: Parsed command line arguments (must have preprocessed_dir and train_test_split_json)
        
    Returns:
        Tuple of (train_loader, val_loader, val_sampler). val_loader may be None
        
    Raises:
        FileNotFoundError: If required directories/files don't exist
    """
    preprocessed_dir = Path(args.preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {args.preprocessed_dir}")
    
    train_test_split_json = Path(args.train_test_split_json)
    if not train_test_split_json.exists():
        raise FileNotFoundError(f"Train/test split file not found: {args.train_test_split_json}")
    
    # Build paths
    patches_dir = preprocessed_dir / "patches"
    patches_info_json = preprocessed_dir / "patches_info.json"
    
    if not patches_dir.exists():
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")
    if not patches_info_json.exists():
        raise FileNotFoundError(f"Patches info JSON not found: {patches_info_json}")
    
    print(f"\nLoading data from: {preprocessed_dir}")
    print(f"  Train/test split: {train_test_split_json}")
    
    # Create training data loader
    print(f"  Loading training data...")
    train_loader, _ = create_dataloader(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        track_indices=False
    )
    
    # Create validation data loader
    print(f"  Loading validation data...")
    val_loader, val_sampler = create_dataloader(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        track_indices=True  # Track indices for validation to get stack_ids and patch_positions
    )
    
    return train_loader, val_loader, val_sampler
