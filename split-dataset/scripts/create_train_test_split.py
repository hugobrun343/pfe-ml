#!/usr/bin/env python3
"""Create a single stratified train/test split."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from split_utils import (
    compute, exclude_stacks_by_id, filter_stacks, format_comparison,
    format_config, format_stats, load_config, load_dataset, plot_distributions,
    print_stats, save_json, stratified_split,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified train/test split.")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    for p, n in [(cfg_path, "Config"), (in_path, "Input")]:
        if not p.exists():
            sys.exit(f"ERROR: {n} not found: {p}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_path.parent / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_path.name

    # Load
    config = load_config(cfg_path)
    dataset = load_dataset(in_path)
    sc = config.get("split", {})
    oc = config.get("output", {})

    # Filter
    stacks = filter_stacks(dataset["stacks"], config.get("filters", {}))
    stacks = exclude_stacks_by_id(stacks, config.get("exclude_stacks") or [])
    if not stacks:
        sys.exit("ERROR: no stacks left.")

    # Split
    train, test = stratified_split(
        stacks, sc.get("test_size", 0.2), sc.get("stratify_by", []), sc.get("random_seed", 42),
    )
    total = len(train) + len(test)
    print(f"Train: {len(train)} ({len(train)/total*100:.1f}%)  Test: {len(test)} ({len(test)/total*100:.1f}%)")

    # Save
    save_json({
        "metadata": {
            "total_samples": total, "train_samples": len(train), "test_samples": len(test),
            "train_ratio": len(train) / total, "test_ratio": len(test) / total,
            "filters_applied": config.get("filters", {}),
            "exclude_stacks": config.get("exclude_stacks") or [],
            "stratification_keys": sc.get("stratify_by", []),
            "random_seed": sc.get("random_seed", 42),
        },
        "train": [s["id"] for s in train],
        "test": [s["id"] for s in test],
    }, out_path, oc.get("indent_json", 2))

    # Stats
    ts_train, ts_test = compute(train), compute(test)
    print_stats(ts_train, "TRAIN")
    print_stats(ts_test, "TEST")

    # Plots
    plot_distributions({"Train": ts_train, "Test": ts_test}, out_dir / "plots")

    # Summary
    if oc.get("generate_summary", True):
        lines = ["TRAIN/TEST SPLIT SUMMARY", f"Generated: {datetime.now().isoformat()}", ""]
        lines += format_config(config) + format_stats(ts_train, "TRAIN")
        lines += format_stats(ts_test, "TEST") + format_comparison(ts_train, ts_test, "Train", "Test")
        summary = out_dir / oc.get("summary_filename", "split_summary.txt")
        summary.write_text("\n".join(lines), encoding="utf-8")
        print(f"Summary: {summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()
