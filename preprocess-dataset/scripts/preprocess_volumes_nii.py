#!/usr/bin/env python3
"""Preprocess 3D medical volumes into patches."""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from config.loader import load_context, get_output_dirs
from run.stacks import load_valid_stacks
from run.pipeline import process_all_volumes
from results.display import print_config_summary, print_run_summary
from results.write import write_run_results
from check import run_post_check


def main():
    parser = argparse.ArgumentParser(description="Preprocess 3D medical volumes into patches")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML configuration file")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--check", action="store_true", help="Run validation on patches at the end")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    ctx = load_context(config_path)
    valid_stacks, total_stacks = load_valid_stacks(ctx["paths"])
    print_config_summary(ctx["cfg"], ctx["norm_config"], ctx["version"])
    print(f"\nValid volumes: {len(valid_stacks)}/{total_stacks}")
    output_base, patches_output = get_output_dirs(ctx)
    print(f"\nOutput: {output_base}")

    n_workers = args.workers if args.workers is not None else mp.cpu_count()
    print(f"\nUsing {n_workers} workers")

    patches_info, volume_count, errors, elapsed, n_ppv = process_all_volumes(ctx, valid_stacks, output_base, patches_output, n_workers)

    write_run_results(ctx, output_base, patches_info, volume_count, n_ppv, errors, elapsed, config_path, n_workers)
    print_run_summary(volume_count, len(valid_stacks), errors, elapsed, output_base, n_ppv)

    if args.check and volume_count > 0:
        cfg = ctx["cfg"]
        expected_shape = (cfg["target_height"], cfg["target_width"], cfg["target_depth"], 3)
        run_post_check(patches_output, volume_count, n_ppv, expected_shape=expected_shape)


if __name__ == "__main__":
    main()
