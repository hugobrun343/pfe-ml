"""Check train/val split integrity."""

from typing import Dict

from .extract import extract_validation_identifiers


def validate(results: dict, split: dict) -> Dict:
    """Ensure no stack or patch appears in both train and validation."""
    train_stacks = set(split.get("train", []))
    test_stacks = set(split.get("test", []))

    val_stacks, val_patches = extract_validation_identifiers(results)

    val_in_train = val_stacks & train_stacks
    val_not_in_test = val_stacks - test_stacks
    ok = len(val_in_train) == 0 and len(val_not_in_test) == 0

    return {
        "ok": ok,
        "train_stacks": len(train_stacks),
        "test_stacks": len(test_stacks),
        "val_unique_stacks": len(val_stacks),
        "val_unique_patches": len(val_patches),
        "val_stacks_in_train": sorted(val_in_train),
        "val_stacks_not_in_test": sorted(val_not_in_test),
    }
