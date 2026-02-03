#!/usr/bin/env python3
"""Rank most problematic stacks across multiple runs. Run: python -m scripts.rank_problematic_stacks [run_dir ...]"""

import sys

from .io import resolve_run_paths
from .stack_ranking import rank_problematic_stacks


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.rank_problematic_stacks <run_dir> [run_dir ...]")
        return 1

    paths = resolve_run_paths(sys.argv[1:], print_skips=True)
    if not paths:
        print("No valid run directories with results found")
        return 1

    print(f"Loading {len(paths)} runs...")
    ranked = rank_problematic_stacks(paths, top_n=25)

    print()
    print("=" * 60)
    print("  MOST PROBLEMATIC STACKS (worst prediction rate across runs)")
    print("=" * 60)
    print(f"  Runs: {len(paths)}")
    print()
    for i, (sid, score, details) in enumerate(ranked, 1):
        pct = score * 100
        print(f"  {i:2}. {sid}  wrong {details['wrong']}/{details['total']} runs ({pct:.0f}%)")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
