"""Print validation report."""

from typing import Dict, List


def _truncate(items: List[str], n: int = 10) -> str:
    return str(items) if len(items) <= n else str(items[:n]) + "..."


def print_report(report: Dict, run_name: str) -> None:
    """Print split validation report."""
    print()
    print("=" * 56)
    print(f"  SPLIT VALIDATION — {run_name}")
    print("=" * 56)
    print(f"  Train: {report['train_stacks']} stacks | Val: {report['test_stacks']} stacks | {report['val_unique_patches']} patches")
    print()
    if report["ok"]:
        print("  ✓ No overlap between train and validation")
    else:
        print("  ✗ Leakage detected")
        if report["val_stacks_in_train"]:
            print(f"    Val stacks in train: {_truncate(report['val_stacks_in_train'])}")
        if report["val_stacks_not_in_test"]:
            print(f"    Val stacks not in test: {_truncate(report['val_stacks_not_in_test'])}")
    print("=" * 56)
    print()
