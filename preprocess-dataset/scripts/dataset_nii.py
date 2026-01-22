#!/usr/bin/env python3
"""
PyTorch Dataset for loading preprocessed .nii.gz patches.
Simple and multiprocessing-safe - each file is independent.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import glob

try:
    import nibabel as nib
except ImportError:
    print("ERROR: pip install nibabel")
    exit(1)


class NIIPatchDataset(Dataset):
    """Dataset that loads preprocessed .nii.gz patch files."""
    
    def __init__(self, patches_dir, metadata_path=None, split='train'):
        """
        Args:
            patches_dir: Directory containing .nii.gz patch files (e.g., output/preprocessed_128x128x32_*/train/)
            metadata_path: Optional path to metadata.json (for getting label from stack_id)
            split: 'train' or 'test' (for metadata lookup)
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        
        # Find all .nii.gz files
        self.patch_files = sorted(list(self.patches_dir.glob("*.nii.gz")))
        
        if len(self.patch_files) == 0:
            raise ValueError(f"No .nii.gz files found in {patches_dir}")
        
        # Load metadata if provided (for labels)
        self.metadata = None
        self.stack_to_label = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            # Build label mapping from filenames
            # Filename format: stack_id_patch_i_j.nii.gz
            for patch_file in self.patch_files:
                stack_id = patch_file.stem.replace('.gz', '').split('_patch_')[0]
                # Try to get label from original dataset if needed
                # For now, we'll extract from filename or use metadata
                pass
        
        print(f"Found {len(self.patch_files)} patches in {patches_dir}")
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        """Load patch from .nii.gz file."""
        patch_file = self.patch_files[idx]
        
        # Load NIfTI file
        img = nib.load(str(patch_file))
        patch = img.get_fdata()
        
        # NIfTI format: (D, H, W, C) = (Z, Y, X, C)
        # Convert to PyTorch format: (C, D, H, W)
        if patch.ndim == 4:  # (D, H, W, C)
            patch = np.transpose(patch, (3, 0, 1, 2))  # → (C, D, H, W)
        elif patch.ndim == 3:  # (D, H, W) - grayscale
            patch = np.expand_dims(patch, axis=0)  # → (1, D, H, W)
        
        # Convert to tensor
        patch = torch.from_numpy(patch.copy()).float()
        
        # Extract label from filename or use default
        # For now, return 0 as placeholder (you can add label extraction logic)
        # Filename: stack_id_patch_i_j.nii.gz
        # You might need to load a separate labels file or extract from metadata
        label = torch.tensor(0, dtype=torch.long)  # TODO: extract actual label
        
        return patch, label


class NIIPatchDatasetWithLabels(Dataset):
    """Dataset with labels and metadata loaded from patches_info.json.
    Filters patches by split using train_test_split.json."""
    
    def __init__(self, patches_dir, patches_info_json, train_test_split_json=None, split='train'):
        """
        Args:
            patches_dir: Directory containing .nii.gz patch files (patches/)
            patches_info_json: Path to patches_info.json file
            train_test_split_json: Path to train_test_split.json file (optional)
            split: 'train' or 'test' to filter patches using train_test_split.json
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        
        # Load patches info
        with open(patches_info_json, 'r') as f:
            all_patches_info = json.load(f)
        
        # Load train/test split if provided
        train_ids = set()
        test_ids = set()
        if train_test_split_json and Path(train_test_split_json).exists():
            with open(train_test_split_json, 'r') as f:
                split_data = json.load(f)
            train_ids = set(split_data.get('train', []))
            test_ids = set(split_data.get('test', []))
            print(f"Loaded split: {len(train_ids)} train, {len(test_ids)} test volumes")
        else:
            print("Warning: No train_test_split.json provided, using all patches")
        
        # Filter patches by split using train_test_split.json
        selected_ids = train_ids if split == 'train' else test_ids
        if not selected_ids:
            # If no split file, use all patches
            self.patches_info = all_patches_info
        else:
            self.patches_info = [p for p in all_patches_info if p['stack_id'] in selected_ids]
        
        # Build file list
        self.patch_files = []
        self.patch_labels = []
        self.patch_metadata = []
        
        for patch_info in self.patches_info:
            patch_file = self.patches_dir / patch_info['filename']
            if patch_file.exists():
                self.patch_files.append(patch_file)
                self.patch_labels.append(patch_info['label'])
                self.patch_metadata.append({
                    'stack_id': patch_info['stack_id'],
                    'position_i': patch_info['position_i'],
                    'position_j': patch_info['position_j']
                })
        
        if len(self.patch_files) == 0:
            raise ValueError(f"No patches found for split '{split}' in {patches_dir}")
        
        print(f"Found {len(self.patch_files)} patches for split '{split}'")
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        """Load patch from .nii.gz file."""
        patch_file = self.patch_files[idx]
        label = self.patch_labels[idx]
        metadata = self.patch_metadata[idx]
        
        # Load NIfTI file
        img = nib.load(str(patch_file))
        patch = img.get_fdata()
        
        # Convert to PyTorch format: (C, D, H, W)
        if patch.ndim == 4:  # (D, H, W, C)
            patch = np.transpose(patch, (3, 0, 1, 2))  # → (C, D, H, W)
        elif patch.ndim == 3:
            patch = np.expand_dims(patch, axis=0)
        
        patch = torch.from_numpy(patch.copy()).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return patch, label
    
    def get_metadata(self, idx):
        """Get metadata for a patch (stack_id, position)."""
        return self.patch_metadata[idx]


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python dataset_nii.py <patches_dir> <patches_info.json> [split]")
        print("Example: python dataset_nii.py output/preprocessed_128x128x32_*/train output/preprocessed_128x128x32_*/patches_info.json train")
        sys.exit(1)
    
    patches_dir = Path(sys.argv[1])
    patches_info_json = Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else 'train'
    
    dataset = NIIPatchDatasetWithLabels(patches_dir, patches_info_json, split=split)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    patch, label = dataset[0]
    metadata = dataset.get_metadata(0)
    print(f"Patch shape: {patch.shape}, Label: {label}")
    print(f"Metadata: {metadata}")
