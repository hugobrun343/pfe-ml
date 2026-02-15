"""Step 3: Define a dataset (NPY patches + JSON train/val splits)."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class JSONSplitPatchDataset(Dataset):
    """Patch dataset using preprocessed NPYs and JSON train/val splits.

    Expected structure in preprocessed_dir:
      - patches_info.json (list of dicts with filename, stack_id, label)
      - patches/ (npy files)

    Expected split JSON keys: train/test. We always use test as val.
    Split values are stack_ids (e.g., "stack_000123").
    """

    def __init__(
        self,
        preprocessed_dir: str,
        splits_json: str,
        split: str,
        channels: int = 1,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.splits_json = Path(splits_json)
        self.split = split
        self.channels = channels

        self._patches = self._load_patches_info()
        self._split_ids = self._load_split_ids()
        self._index = self._filter_by_split()

    def _load_patches_info(self) -> List[Dict]:
        info_path = self.preprocessed_dir / "patches_info.json"
        return json.loads(info_path.read_text())

    def _load_split_ids(self) -> List[str]:
        split_data = json.loads(self.splits_json.read_text())

        if self.split == "val":
            if "test" not in split_data:
                raise KeyError(f"Split 'test' not found in {self.splits_json}")
            ids = split_data["test"]
        elif self.split == "train":
            if "train" not in split_data:
                raise KeyError(f"Split 'train' not found in {self.splits_json}")
            ids = split_data["train"]
        else:
            raise KeyError(f"Unsupported split '{self.split}'. Use 'train' or 'val'.")

        return ids

    def _filter_by_split(self) -> List[int]:
        split_set = set(self._split_ids)
        return [i for i, p in enumerate(self._patches) if p["stack_id"] in split_set]

    def __len__(self) -> int:
        return len(self._index)

    def _load_patch(self, filename: str) -> torch.Tensor:
        import numpy as np

        path = self.preprocessed_dir / "patches" / filename
        arr = np.load(path)
        x = torch.as_tensor(arr, dtype=torch.float32)

        # Ensure channel-first (C,D,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (D,H,W) -> (1,D,H,W)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected patch shape: {x.shape}")

        # Select a single channel (1-indexed: channel=1 -> index 0, etc.)
        if x.shape[0] > 1:
            ch_idx = self.channels - 1  # convert 1-indexed to 0-indexed
            if ch_idx < 0 or ch_idx >= x.shape[0]:
                raise ValueError(
                    f"Channel {self.channels} out of range for patch with {x.shape[0]} channels"
                )
            x = x[ch_idx : ch_idx + 1]  # (1, D, H, W)

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self._patches[self._index[idx]]
        x = self._load_patch(p["filename"])
        y = torch.tensor([float(p["label"])], dtype=torch.float32)
        return x, y


def make_train_val_loaders(
    preprocessed_dir: str,
    splits_json: str,
    batch_size: int,
    channels: int = 1,
):
    train_set = JSONSplitPatchDataset(
        preprocessed_dir=preprocessed_dir,
        splits_json=splits_json,
        split="train",
        channels=channels,
    )
    val_set = JSONSplitPatchDataset(
        preprocessed_dir=preprocessed_dir,
        splits_json=splits_json,
        split="val",
        channels=channels,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
    )

    return train_loader, val_loader
