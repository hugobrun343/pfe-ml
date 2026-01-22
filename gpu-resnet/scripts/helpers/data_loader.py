"""Data loader utilities for ResNet3D training"""

from torch.utils.data import DataLoader, Sampler
from typing import Optional, Tuple, Iterator
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "helpers"))

from scripts.helpers.dataset import NIIPatchDataset


class IndexTrackingSampler(Sampler):
    """Sampler that tracks indices for validation (shuffle=False)"""
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.indices = list(range(len(data_source)))
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)


def create_dataloader(
    patches_dir: str,
    patches_info_json: str,
    train_test_split_json: str,
    split: str = 'train',
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[callable] = None,
    prefetch_factor: int = 2,
    track_indices: bool = False
) -> Tuple[DataLoader, Optional[IndexTrackingSampler]]:
    """
    Create DataLoader for .nii.gz patches
    
    
    Args:
        patches_dir: Directory containing .nii.gz patch files
        patches_info_json: Path to patches_info.json file
        train_test_split_json: Path to train_test_split.json file (REQUIRED)
        split: 'train' or 'test' to filter patches
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (each loads files independently)
        pin_memory: Whether to pin memory for faster GPU transfer
        transform: Optional transform to apply
        prefetch_factor: Number of batches each worker prefetches (default: 2)
        track_indices: Whether to track indices (for validation)
        
    Returns:
        Tuple of (DataLoader, Optional[IndexTrackingSampler])
    """
    # Create dataset
    dataset = NIIPatchDataset(
        patches_dir=patches_dir,
        patches_info_json=patches_info_json,
        train_test_split_json=train_test_split_json,
        split=split,
        transform=transform
    )
    
    # Use custom sampler to track indices if requested (for validation)
    sampler = None
    if track_indices and not shuffle:
        sampler = IndexTrackingSampler(dataset, batch_size)
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        timeout=60  # Timeout for worker processes (60 seconds)
    )
    
    return dataloader, sampler


def get_sample_metadata(
    dataset,
    indices: list
) -> Tuple[list, list]:
    """
    Get stack_ids and grid positions for samples from dataset
    
    Args:
        dataset: NIIPatchDataset instance
        indices: List of indices in the dataset
        
    Returns:
        Tuple of (stack_ids, grid_positions) where:
        - stack_ids: List of volume IDs for each sample
        - grid_positions: List of (i, j) tuples for each sample's position in the grid
    """
    stack_ids = []
    grid_positions = []
    
    for idx in indices:
        metadata = dataset.get_metadata(idx)
        stack_ids.append(metadata['stack_id'])
        grid_positions.append((metadata['position_i'], metadata['position_j']))
    
    return stack_ids, grid_positions
