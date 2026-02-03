#!/usr/bin/env python3
"""Report patch vs aggregated F1. Run: python -m scripts.report_aggregated_f1 [run_dir ...]"""

import sys

from .io import resolve_run_paths
from .aggregated_metrics import analyze_runs
from .report import print_aggregated_f1_report


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.report_aggregated_f1 <run_dir> [run_dir ...]")
        return 1
    paths = resolve_run_paths(sys.argv[1:])
    if not paths:
        print("No valid run directories with results found")
        return 1
    best, _ = analyze_runs(paths)
    if not best:
        print("No runs with validation/volume data found")
        return 1
    print_aggregated_f1_report(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
