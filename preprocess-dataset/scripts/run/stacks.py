"""Load and filter valid stacks from dataset."""

import json
from pathlib import Path
from typing import List, Dict, Tuple


def _filter_valid_stacks(dataset: dict, data_root: Path) -> List[dict]:
    """Keep only SAIN/MALADE stacks with existing nii_path."""
    valid = []
    for stack in dataset["stacks"]:
        if stack.get("infos", {}).get("Classe", "") not in ("SAIN", "MALADE"):
            continue
        if not stack.get("nii_path"):
            continue
        vol_path = data_root / Path(stack["nii_path"]).name.replace(".nii", ".nii.gz")
        if vol_path.exists():
            valid.append(stack)
    return valid


def load_valid_stacks(paths: Dict[str, Path]) -> Tuple[List[dict], int]:
    """Load dataset JSON and return (valid_stacks, total_stacks)."""
    with open(paths["dataset_json"], "r") as f:
        dataset = json.load(f)
    valid = _filter_valid_stacks(dataset, paths["data_root"])
    return valid, len(dataset["stacks"])
