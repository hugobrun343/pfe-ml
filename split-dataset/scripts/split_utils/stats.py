"""Compute and display distribution statistics."""

from collections import Counter
from typing import Any, Dict, List

CATEGORIES = [
    "age", "sex", "orientation", "genetic",
    "classe", "pressure", "axial_stretch", "region",
]

_DS_KEY = {
    "age": "Age (wk)",  "sex": "Sex",  "orientation": "Orientation",
    "genetic": "Genetic",  "classe": "Classe",  "pressure": "Pressure",
    "axial_stretch": "Axial Stretch",  "region": "Region ",
}


def compute(stacks: List[Dict]) -> Dict[str, Any]:
    """Return ``{total, age: Counter, sex: Counter, ...}``."""
    stats: Dict[str, Any] = {"total": len(stacks)}
    for cat in CATEGORIES:
        stats[cat] = Counter()
    for stack in stacks:
        infos = stack.get("infos", {})
        for cat in CATEGORIES:
            raw = infos.get(_DS_KEY[cat], "Unknown")
            stats[cat][raw.strip() if isinstance(raw, str) else raw] += 1
    return stats


def pretty_name(cat: str) -> str:
    """Human-friendly category label."""
    return "Axial Stretch" if cat == "axial_stretch" else cat.replace("_", " ").title()


def print_stats(stats: Dict, label: str) -> None:
    """Print distribution to stdout."""
    print(f"\n{'=' * 70}")
    print(f"{label}  ({stats['total']} samples)")
    print("=" * 70)
    for cat in CATEGORIES:
        if not stats[cat]:
            continue
        print(f"  {pretty_name(cat)}:")
        for val, cnt in sorted(stats[cat].items(), key=lambda x: -x[1]):
            print(f"    {str(val):28s} {cnt:4d} ({cnt / stats['total'] * 100:5.1f}%)")
