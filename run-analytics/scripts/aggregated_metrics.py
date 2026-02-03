"""Volume-level (aggregated) metrics analysis."""

from typing import Dict, List, Optional, Tuple

from .extract import get_validation_metrics
from .io import load_results


def find_best_epoch(results: dict) -> Optional[int]:
    """Epoch with highest validation f1_class_1 (patch-level)."""
    best_epoch, best_f1 = None, -1.0
    for e in results.get("epochs", []):
        f1 = e.get("validation", {}).get("f1_class_1")
        if f1 is not None and f1 > best_f1:
            best_f1, best_epoch = f1, e.get("epoch")
    return best_epoch


def get_run_summary(results: dict, run_path: str = "") -> Optional[Dict]:
    """Summary: best epoch, patch and volume metrics."""
    best_epoch = find_best_epoch(results)
    if best_epoch is None:
        return None
    patch, volume = get_validation_metrics(results, best_epoch)
    if patch is None:
        return None
    return {
        "run_name": results.get("training_config", {}).get("run_name", run_path),
        "run_path": run_path,
        "best_epoch": best_epoch,
        "patch_metrics": patch,
        "volume_metrics": volume or {},
        "best_val_f1_class_1": results.get("best_val_f1_class_1"),
    }


def analyze_runs(run_paths: List[str]) -> Tuple[Optional[Dict], List[Dict]]:
    """Analyze runs, return (best_summary, all_summaries)."""
    summaries = []
    for path in run_paths:
        try:
            s = get_run_summary(load_results(path), run_path=path)
        except FileNotFoundError:
            continue
        if s:
            summaries.append(s)
    if not summaries:
        return None, []
    best = max(summaries, key=lambda x: x.get("best_val_f1_class_1") or x["patch_metrics"].get("f1_class_1") or 0)
    return best, summaries
