#!/usr/bin/env python3
"""
Test VRAM for 256x256x32x3 — ResNet3D, SE-ResNet3D, ViT3D, ConvNeXt3D.

Runs each model family in a separate subprocess to avoid GPU cascade.
Output: results/test_256x256x32.json
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from test_256x256x32 import DEFAULT_FAMILIES, run_all_families, run_single_family


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="VRAM test 256x256x32x3. Each family runs in a separate process."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_FAMILIES,
        choices=DEFAULT_FAMILIES,
        help="Model families to test (default: all)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path (subprocess use)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else ROOT / "results" / "test_256x256x32.json"

    if len(args.models) == 1 or args.output:
        return run_single_family(args.models, out_path)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    return run_all_families(args.models, SCRIPTS / "test_256x256x32.py", ROOT, results_dir)


if __name__ == "__main__":
    sys.exit(main())
