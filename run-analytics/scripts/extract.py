"""Extract identifiers from validation samples."""

from typing import Set, Tuple


def extract_validation_identifiers(
    results: dict,
    last_epoch_only: bool = True,
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
