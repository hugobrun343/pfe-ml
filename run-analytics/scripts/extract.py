"""Extract data from training results."""

from typing import Dict, Optional, Set, Tuple


def get_epoch(results: dict, epoch_num: int) -> Optional[Dict]:
    """Get epoch dict by number."""
    for e in results.get("epochs", []):
        if e.get("epoch") == epoch_num:
            return e
    return None


def get_validation_metrics(results: dict, epoch_num: int) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Get (patch_metrics, volume_metrics) for an epoch."""
    ep = get_epoch(results, epoch_num)
    if not ep:
        return None, None
    val = ep.get("validation", {})
    patch = {"f1_class_0": val.get("f1_class_0"), "f1_class_1": val.get("f1_class_1"), "accuracy": val.get("accuracy")} if val else None
    volume = val.get("volume_metrics")
    return patch, volume


def extract_validation_identifiers(
    results: dict,
    last_epoch_only: bool = False,
) -> Tuple[Set[str], Set[Tuple]]:
    """Extract stack_ids and (stack_id, position_h, position_w, patch_index) from validation samples."""
    epochs = results.get("epochs", [])
    if not epochs:
        return set(), set()

    if last_epoch_only:
        epochs = [epochs[-1]]

    stacks = set()
    patches = set()
    for ep in epochs:
        samples = ep.get("validation", {}).get("samples", [])
        for s in samples:
            sid = s.get("stack_id")
            if sid:
                stacks.add(sid)
                ph, pw, pi = s.get("position_h"), s.get("position_w"), s.get("patch_index")
                if ph is not None and pw is not None:
                    patches.add((sid, ph, pw, pi))

    return stacks, patches
