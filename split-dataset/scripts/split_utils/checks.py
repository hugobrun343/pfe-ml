"""Isolation and distribution checks for cross-validation splits."""

from typing import Dict, List, Set

from .stats import CATEGORIES, pretty_name


def run_isolation_checks(
    test_ids: Set[str],
    fold_val_ids: List[Set[str]],
    fold_train_ids: List[Set[str]],
    all_cv_ids: Set[str],
) -> List[str]:
    """Verify no data leaks between test / val / train."""
    n = len(fold_val_ids)
    lines: List[str] = ["=" * 70, "ISOLATION CHECKS", "=" * 70]
    ok = True

    def _chk(name: str, passed: bool, detail: str = ""):
        nonlocal ok
        if not passed:
            ok = False
        tag = "PASS" if passed else "FAIL"
        lines.append(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))

    for i in range(n):
        leak = test_ids & (fold_train_ids[i] | fold_val_ids[i])
        _chk(f"Test hold-out vs fold {i}", not leak, f"{len(leak)} leaks" if leak else "")

    for i in range(n):
        for j in range(i + 1, n):
            ov = fold_val_ids[i] & fold_val_ids[j]
            _chk(f"Val {i} vs val {j} disjoint", not ov, f"{len(ov)} overlap" if ov else "")

    union = set().union(*fold_val_ids)
    _chk("Union(val folds) == CV pool", union == all_cv_ids,
         f"union={len(union)}, expected={len(all_cv_ids)}")

    for i in range(n):
        ov = fold_train_ids[i] & fold_val_ids[i]
        _chk(f"Train/val disjoint fold {i}", not ov, f"{len(ov)} overlap" if ov else "")

    sizes = [len(ids) for ids in fold_val_ids]
    lines += ["", "FOLD SIZES", "-" * 40]
    for i, s in enumerate(sizes):
        lines.append(f"  Fold {i} val: {s}")
    mean = sum(sizes) / len(sizes)
    dev = max(abs(s - mean) for s in sizes)
    _chk("Val sizes balanced", dev <= max(mean * 0.15, 2),
         f"mean={mean:.1f}, max_dev={dev:.1f}")

    lines += ["", "=" * 70, f"{'ALL PASSED' if ok else 'SOME FAILED'}", "=" * 70]
    return lines


def run_distribution_checks(
    global_stats: Dict,
    fold_stats: List[Dict],
) -> List[str]:
    """Flag per-fold distribution deviations > 5 % vs the CV pool."""
    n = len(fold_stats)
    lines = ["", "=" * 70, "DISTRIBUTION CHECKS  (>5% flagged)", "=" * 70]
    warnings = 0

    for cat in CATEGORIES:
        lines.append(f"\n  {pretty_name(cat)}:")
        vals = set(global_stats[cat])
        for fs in fold_stats:
            vals |= set(fs[cat])
        for v in sorted(vals):
            gp = global_stats[cat].get(v, 0) / max(global_stats["total"], 1) * 100
            fps = [fs[cat].get(v, 0) / max(fs["total"], 1) * 100 for fs in fold_stats]
            md = max(abs(p - gp) for p in fps)
            flag = ""
            if md > 5:
                flag = " ***"
                warnings += 1
            cols = "  ".join(f"F{i}:{p:5.1f}%" for i, p in enumerate(fps))
            lines.append(f"    {str(v):22s} global {gp:5.1f}%  {cols}  diff {md:4.1f}%{flag}")

    lines += ["", f"  Warnings (>5%): {warnings}", ""]
    return lines
