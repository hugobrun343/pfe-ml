"""Load test holdout patches from cv_global.json + patches_info.json."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TestHoldoutDataset(Dataset):
    """Dataset of patches belonging to the test holdout set.

    Returns (tensor, label, patch_filename, stack_id) for each patch.
    """

    def __init__(self, preprocessed_dir: str, cv_global_json: str):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.patches_dir = self.preprocessed_dir / "patches"

        # Load holdout stack IDs
        with open(cv_global_json, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        holdout_ids = set(cv_data["test_holdout"])

        # Load all patch metadata and filter to holdout
        info_path = self.preprocessed_dir / "patches_info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            all_patches = json.load(f)

        self.patches: List[Dict] = [
            p for p in all_patches if p["stack_id"] in holdout_ids
        ]

        if len(self.patches) == 0:
            raise RuntimeError(
                f"No patches found for test holdout. "
                f"Holdout has {len(holdout_ids)} stacks, "
                f"patches_info has {len(all_patches)} entries."
            )

    def __len__(self) -> int:
        return len(self.patches)

    def _load_patch(self, filename: str) -> torch.Tensor:
        arr = np.load(self.patches_dir / filename)
        x = torch.as_tensor(arr, dtype=torch.float32)

        # Ensure channel-first (C, D, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Single channel -> repeat to 3 channels
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1, 1)

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, str, str]:
        p = self.patches[idx]
        tensor = self._load_patch(p["filename"])
        label = float(p["label"])
        return tensor, label, p["filename"], p["stack_id"]


def make_test_loader(
    preprocessed_dir: str,
    cv_global_json: str,
    batch_size: int,
) -> DataLoader:
    """Create a DataLoader for the test holdout set."""
    dataset = TestHoldoutDataset(preprocessed_dir, cv_global_json)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
