"""Dataset for ResNet3D training"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from pathlib import Path
import json
import numpy as np
import nibabel as nib



class NIIPatchDataset(Dataset):
    """Dataset for 3D patches loaded from .nii.gz files.
    
    Simple and multiprocessing-safe - each file is independent.
    Filters patches using train_test_split.json and patches_info.json.
    """
    
    def __init__(
        self,
        patches_dir: str,
        patches_info_json: str,
        train_test_split_json: str,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        """
        Args:
            patches_dir: Directory containing .nii.gz patch files (patches/)
            patches_info_json: Path to patches_info.json file
            train_test_split_json: Path to train_test_split.json file (REQUIRED)
            split: 'train' or 'test' to filter patches using train_test_split.json
            transform: Optional transform to apply
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        self.transform = transform
        
        # Load patches info
        with open(patches_info_json, 'r') as f:
            all_patches_info = json.load(f)
        
        # Load train/test split - REQUIRED
        if not train_test_split_json:
            raise ValueError("train_test_split_json is required")
        
        train_test_split_path = Path(train_test_split_json)
        if not train_test_split_path.exists():
            raise FileNotFoundError(f"Train/test split file not found: {train_test_split_json}")
        
        with open(train_test_split_path, 'r') as f:
            split_data = json.load(f)
        
        # Validate split file format
        if 'train' not in split_data or 'test' not in split_data:
            raise ValueError(f"Invalid train_test_split.json format: missing 'train' or 'test' keys")
        
        train_ids = set(split_data['train'])
        test_ids = set(split_data['test'])
        
        # Validate split parameter
        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")
        
        print(f"Loaded split: {len(train_ids)} train, {len(test_ids)} test volumes")
        
        # Filter patches by split using train_test_split.json
        selected_ids = train_ids if split == 'train' else test_ids
        if not selected_ids:
            raise ValueError(f"No volumes found in split '{split}' from train_test_split.json")
        
        self.patches_info = [p for p in all_patches_info if p['stack_id'] in selected_ids]
        
        if not self.patches_info:
            raise ValueError(f"No patches found for split '{split}' after filtering by train_test_split.json")
        
        # Build file list and metadata
        self.patch_files = []
        self.patch_labels = []
        self.patch_metadata = []
        
        for patch_info in self.patches_info:
            patch_file = self.patches_dir / patch_info['filename']
            if not patch_file.exists():
                raise FileNotFoundError(f"Patch file not found: {patch_file}")
            
            self.patch_files.append(patch_file)
            self.patch_labels.append(patch_info['label'])
            
            if 'position_h' not in patch_info or 'position_w' not in patch_info:
                raise ValueError(f"Patch info missing position_h or position_w: {patch_info.keys()}")
            
            self.patch_metadata.append({
                'stack_id': patch_info['stack_id'],
                'position_h': patch_info['position_h'],
                'position_w': patch_info['position_w'],
                'patch_index': patch_info.get('patch_index', None)
            })
        
        if len(self.patch_files) == 0:
            raise ValueError(f"No patches found for split '{split}' in {patches_dir}")
        
        print(f"Found {len(self.patch_files)} patches for split '{split}'")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset - loads .nii.gz file on-demand
        
        Optimized loading: use mmap_mode for faster access and avoid unnecessary copies.

        Args:
            idx: Index of item
            
        Returns:
            Tuple of (block, label) where block is (3, D, H, W) and label is float32
        """
        try:
            patch_file = self.patch_files[idx]
            label = self.patch_labels[idx]
            
            # Load NIfTI file - optimized: use direct array access
            img = nib.load(str(patch_file))
            patch = np.asarray(img.dataobj, dtype=np.float32)
            
            # Convert to PyTorch format: (C, D, H, W)
            # NIfTI format: (D, H, W, C) = (Z, Y, X, C)
            if patch.ndim == 4:  # (D, H, W, C)
                patch = np.transpose(patch, (3, 0, 1, 2))  # → (C, D, H, W)
            elif patch.ndim == 3:
                patch = np.expand_dims(patch, axis=0)  # → (1, D, H, W)
            
            # Ensure correct types (already float32 from asarray)
            if patch.dtype != np.float32:
                patch = patch.astype(np.float32, copy=False)
            
            # Convert to tensor - use from_numpy directly (no copy needed if contiguous)
            patch = torch.from_numpy(patch).float()
            label = torch.tensor(label, dtype=torch.float32)
            
            if self.transform:
                patch = self.transform(patch)
            
            return patch, label
        except Exception as e:
            # Log error with file info for debugging
            import sys
            print(f"ERROR loading patch {idx} from {patch_file}: {e}", file=sys.stderr, flush=True)
            raise
    
    def get_metadata(self, idx: int) -> dict:
        """
        Get metadata for a patch.
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary with 'stack_id', 'position_h', 'position_w', 'patch_index'
        """
        return self.patch_metadata[idx]
