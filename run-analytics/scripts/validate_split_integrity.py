#!/usr/bin/env python3
"""Validate no stack or patch appears in both train and validation. Run: python -m scripts.validate_split_integrity <run_dir>"""

import sys
from pathlib import Path

from .io import load_run
from .validate import validate
from .report import print_report


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.validate_split_integrity <run_dir>")
        return 1

    run_path = Path(sys.argv[1])
    if not run_path.exists():
        print(f"Error: {run_path} not found")
        return 1

    try:
        results, split = load_run(run_path)
    except FileNotFoundError as e:
        print(e)
        return 1

    report = validate(results, split)
    print_report(report, run_path.name)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
