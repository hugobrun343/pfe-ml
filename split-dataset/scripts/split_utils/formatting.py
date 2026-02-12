"""Text formatting helpers for summary files."""

from typing import Dict, List

from .stats import CATEGORIES, pretty_name


def format_config(config: Dict) -> List[str]:
    """Lines describing filters, split params, exclusions."""
    filters = config.get("filters", {})
    split = config.get("split", {})
    exclude = config.get("exclude_stacks") or []

    lines = ["CONFIGURATION", "=" * 70, "", "Filters:"]
    age = filters.get("age")
    if age is not None:
        lines.append(f"  Age (wk): min={age.get('min')}, max={age.get('max')}")
    for k in ["axial_stretch", "pressure", "region", "classe", "genetic", "sex", "orientation"]:
        v = filters.get(k)
        if v is not None:
            lines.append(f"  {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    if exclude:
        lines.append(f"  Excluded: {', '.join(str(s) for s in exclude)}")

    lines += ["", "Split:"]
    lines.append(f"  test_size: {split.get('test_size', 0.2)}")
    if "n_folds" in split:
        lines.append(f"  n_folds:   {split['n_folds']}")
    lines.append(f"  seed:      {split.get('random_seed', 42)}")
    lines.append(f"  stratify:  {', '.join(split.get('stratify_by', []))}")
    lines.append("")
    return lines


def format_stats(stats: Dict, label: str) -> List[str]:
    """Distribution table for one set."""
    lines = ["=" * 70, f"{label}  ({stats['total']} samples)", "=" * 70]
    for cat in CATEGORIES:
        if not stats[cat]:
            continue
        lines.append(f"\n  {pretty_name(cat)}:")
        for val, cnt in sorted(stats[cat].items(), key=lambda x: -x[1]):
            lines.append(f"    {str(val):28s} {cnt:4d} ({cnt / stats['total'] * 100:5.1f}%)")
    lines.append("")
    return lines


def format_comparison(a: Dict, b: Dict, la: str, lb: str) -> List[str]:
    """Side-by-side comparison of two stat dicts."""
    lines = ["=" * 70, f"COMPARISON  {la} vs {lb}", "=" * 70]
    for cat in CATEGORIES:
        lines.append(f"\n  {pretty_name(cat)}:")
        vals = sorted(set(a[cat]) | set(b[cat]))
        for v in vals:
            pa = a[cat].get(v, 0) / max(a["total"], 1) * 100
            pb = b[cat].get(v, 0) / max(b["total"], 1) * 100
            d = abs(pa - pb)
            flag = " ***" if d > 5 else ""
            lines.append(f"    {str(v):28s} {la} {pa:5.1f}% | {lb} {pb:5.1f}% | diff {d:4.1f}%{flag}")
    lines.append("")
    return lines
