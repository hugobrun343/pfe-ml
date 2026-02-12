#!/usr/bin/env python3
"""Create a stratified k-fold CV split with a hold-out test set."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from split_utils import (
    compute, exclude_stacks_by_id, filter_stacks, format_comparison,
    format_config, format_stats, load_config, load_dataset, plot_distributions,
    print_stats, run_distribution_checks, run_isolation_checks, save_json,
    stratified_kfold, stratified_split,
)


# -- JSON builders -----------------------------------------------------------

def _fold_json(train, val, idx, n, config):
    total = len(train) + len(val)
    return {
        "metadata": {
            "total_samples": total, "train_samples": len(train), "test_samples": len(val),
            "train_ratio": len(train) / total, "test_ratio": len(val) / total,
            "fold": idx, "n_folds": n,
            "filters_applied": config.get("filters", {}),
            "exclude_stacks": config.get("exclude_stacks") or [],
            "stratification_keys": config["split"].get("stratify_by", []),
            "random_seed": config["split"].get("random_seed", 42),
        },
        "train": [s["id"] for s in train],
        "test": [s["id"] for s in val],
    }


def _global_json(holdout, folds, config):
    n_cv = sum(len(f) for f in folds)
    return {
        "metadata": {
            "total_samples": len(holdout) + n_cv,
            "test_holdout_samples": len(holdout), "cv_samples": n_cv,
            "n_folds": len(folds),
            "filters_applied": config.get("filters", {}),
            "exclude_stacks": config.get("exclude_stacks") or [],
            "stratification_keys": config["split"].get("stratify_by", []),
            "random_seed": config["split"].get("random_seed", 42),
        },
        "test_holdout": [s["id"] for s in holdout],
        "folds": [
            {"fold": i,
             "val": [s["id"] for s in folds[i]],
             "train": [s["id"] for j, f in enumerate(folds) for s in f if j != i]}
            for i in range(len(folds))
        ],
    }


# -- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified k-fold CV split.")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output-dir", "-o", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    in_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()

    for p, n in [(cfg_path, "Config"), (in_path, "Input")]:
        if not p.exists():
            sys.exit(f"ERROR: {n} not found: {p}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(cfg_path)
    dataset = load_dataset(in_path)
    sc = config["split"]
    indent = config.get("output", {}).get("indent_json", 2)

    # Filter
    stacks = filter_stacks(dataset["stacks"], config.get("filters", {}))
    stacks = exclude_stacks_by_id(stacks, config.get("exclude_stacks") or [])
    if not stacks:
        sys.exit("ERROR: no stacks left.")

    # Hold-out test
    n_folds = sc.get("n_folds", 5)
    seed = sc.get("random_seed", 42)
    strat = sc.get("stratify_by", [])
    cv_pool, holdout = stratified_split(stacks, sc.get("test_size", 0.1), strat, seed)
    print(f"Hold-out test: {len(holdout)}  |  CV pool: {len(cv_pool)}")

    # K-fold
    folds = stratified_kfold(cv_pool, n_folds, strat, seed + 1)
    for i, f in enumerate(folds):
        print(f"  Fold {i}: {len(f)} samples")

    # Save per-fold JSONs
    for i in range(n_folds):
        train = [s for j in range(n_folds) if j != i for s in folds[j]]
        save_json(_fold_json(train, folds[i], i, n_folds, config),
                  out_dir / f"train_test_split_fold_{i}.json", indent)

    save_json(_global_json(holdout, folds, config), out_dir / "cv_global.json", indent)

    # Stats
    ho_stats = compute(holdout)
    pool_stats = compute(cv_pool)
    fold_stats = [compute(f) for f in folds]
    train_sets = [[s for j in range(n_folds) if j != i for s in folds[j]] for i in range(n_folds)]
    train_stats = [compute(ts) for ts in train_sets]
    print_stats(ho_stats, "TEST HOLD-OUT")
    for i in range(n_folds):
        print_stats(fold_stats[i], f"FOLD {i} VAL")

    # Plots
    val_dict = {f"Fold {i} val": fold_stats[i] for i in range(n_folds)}
    trn_dict = {f"Fold {i} train": train_stats[i] for i in range(n_folds)}
    plot_distributions(val_dict, out_dir / "plots_val")
    plot_distributions(trn_dict, out_dir / "plots_train")
    plot_distributions(
        {"Test hold-out": ho_stats}, out_dir / "plots_holdout",
        stats_dict_pct={"Test hold-out": ho_stats, "CV pool": pool_stats},
    )

    # Checks
    test_ids = {s["id"] for s in holdout}
    fv_ids = [{s["id"] for s in f} for f in folds]
    ft_ids = [{s["id"] for j in range(n_folds) if j != i for s in folds[j]} for i in range(n_folds)]
    cv_ids = {s["id"] for s in cv_pool}

    iso = run_isolation_checks(test_ids, fv_ids, ft_ids, cv_ids)
    dist = run_distribution_checks(pool_stats, fold_stats)
    for line in iso + dist:
        print(line)

    if config.get("output", {}).get("generate_checks", True):
        (out_dir / "cv_checks.txt").write_text("\n".join(iso + dist), encoding="utf-8")

    if config.get("output", {}).get("generate_summary", True):
        lines = ["CV SPLIT SUMMARY", f"Generated: {datetime.now().isoformat()}", ""]
        lines += format_config(config) + format_stats(ho_stats, "TEST HOLD-OUT")
        lines += format_stats(pool_stats, "CV POOL")
        for i in range(n_folds):
            lines += format_stats(fold_stats[i], f"FOLD {i} VAL")
        lines += format_comparison(ho_stats, pool_stats, "Test", "CV pool")
        (out_dir / "cv_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")


if __name__ == "__main__":
    main()
