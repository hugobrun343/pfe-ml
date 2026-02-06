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

from scripts.helpers.dataset import NIIPatchDataset


def load_data(args: Namespace):
    """
    Load training and validation data as separate DataLoaders.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
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
    
    # Create datasets
    train_dataset = NIIPatchDataset(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split='train',
        transform=None
    )
    
    val_dataset = NIIPatchDataset(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split='test',
        transform=None
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Train DataLoader - shuffle enabled
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        drop_last=False
    )
    
    # Val DataLoader - no shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        drop_last=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset
