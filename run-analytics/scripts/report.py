"""Print reports."""

from typing import Dict, List, Optional


def format_metrics(m: Optional[Dict]) -> str:
    """Format metrics dict for display."""
    if not m:
        return "  (n/a)"
    labels = {"f1_class_0": "F1 class 0", "f1_class_1": "F1 class 1", "accuracy": "accuracy"}
    parts = []
    for k in ("f1_class_0", "f1_class_1", "accuracy"):
        v = m.get(k)
        if v is not None:
            parts.append(f"{labels.get(k, k)}: {v:.4f}")
    return "  " + " | ".join(parts) if parts else "  (n/a)"


def print_aggregated_f1_report(best: Dict) -> None:
    """Print aggregated F1 report for best run."""
    pm = best.get("patch_metrics") or {}
    vm = best.get("volume_metrics") or {}
    print()
    print("=" * 70)
    print("  AGGREGATED (VOLUME-LEVEL) F1 — BEST RUN")
    print("=" * 70)
    print(f"  Best run (by patch f1_class_1): {best['run_name']}")
    print(f"  Best epoch: {best['best_epoch']}")
    print()
    print("  Patch-level (validation):")
    print(format_metrics(pm))
    print()
    print("  Aggregated (volume-level, mean prob > 0.5):")
    print(format_metrics(vm))
    print("=" * 70)
    print()


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
