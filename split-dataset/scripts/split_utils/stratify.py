"""Stratified splitting and k-fold partitioning."""

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def _stratification_key(stack: Dict, fields: List[str]) -> str:
    """Composite key from metadata *fields*."""
    infos = stack.get("infos", {})
    return "_".join(str(infos.get(f, "Unknown")) for f in fields)


def stratified_split(
    stacks: List[Dict],
    test_size: float,
    stratify_by: List[str],
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Two-way stratified split -> (larger, smaller ≈ test_size)."""
    random.seed(seed)
    strata: Dict[str, List[Dict]] = defaultdict(list)
    for s in stacks:
        strata[_stratification_key(s, stratify_by)].append(s)

    larger, smaller = [], []
    for key in sorted(strata):
        group = strata[key]
        random.shuffle(group)
        n = max(1, int(len(group) * test_size))
        if len(group) == 1:
            larger.extend(group)
        else:
            smaller.extend(group[:n])
            larger.extend(group[n:])
    return larger, smaller


def stratified_kfold(
    stacks: List[Dict],
    n_folds: int,
    stratify_by: List[str],
    seed: int,
) -> List[List[Dict]]:
    """Round-robin over strata -> *n_folds* roughly equal buckets."""
    random.seed(seed)
    strata: Dict[str, List[Dict]] = defaultdict(list)
    for s in stacks:
        strata[_stratification_key(s, stratify_by)].append(s)

    folds: List[List[Dict]] = [[] for _ in range(n_folds)]
    for key in sorted(strata):
        group = strata[key]
        random.shuffle(group)
        offset = random.randint(0, n_folds - 1)
        for i, s in enumerate(group):
            folds[(i + offset) % n_folds].append(s)
    return folds
