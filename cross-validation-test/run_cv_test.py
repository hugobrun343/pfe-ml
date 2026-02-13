#!/usr/bin/env python3
"""Cross-validation test pipeline.

Loads 5 fold checkpoints, runs inference on the test holdout set,
aggregates patch scores into stack scores, ensembles the 5 models,
and outputs detailed results.

Usage:
    python run_cv_test.py --config configs/test_cv_resnet3d_50.yaml
"""

import argparse
import time
from pathlib import Path

from cv_test.config import load_config
from cv_test.dataset import make_test_loader
from cv_test.inference import run_inference
from cv_test.aggregate import aggregate_patches_to_stacks, ensemble_stacks
from cv_test.metrics import compute_metrics
from cv_test.report import save_results_json, save_summary_txt


def main():
    parser = argparse.ArgumentParser(description="CV Test Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # ── 1. Load config ────────────────────────────────────────────
    print(f"[1/6] Loading config: {args.config}")
    cfg = load_config(args.config)
    cfg.validate()

    # ── 2. Load test holdout data ─────────────────────────────────
    print(f"[2/6] Loading test holdout patches...")
    loader = make_test_loader(
        preprocessed_dir=cfg.preprocessed_dir,
        cv_global_json=cfg.cv_global_json,
        batch_size=cfg.batch_size,
    )
    n_patches = len(loader.dataset)
    print(f"       {n_patches} patches loaded")

    # Build patch -> stack mapping
    patch_to_stack = {
        p["filename"]: p["stack_id"] for p in loader.dataset.patches
    }
    # Build stack -> label mapping
    stack_labels = {
        p["stack_id"]: int(p["label"]) for p in loader.dataset.patches
    }

    # ── 3. Run inference on each fold ─────────────────────────────
    all_patch_scores = {}   # fold_key -> {filename: prob}
    all_stack_scores = {}   # fold_key -> {stack_id: mean_prob}

    for fold_idx, ckpt_path in enumerate(cfg.checkpoints):
        fold_key = f"fold_{fold_idx}"
        print(f"[3/6] Inference {fold_key}: {Path(ckpt_path).name}...")
        t0 = time.time()

        patch_scores = run_inference(
            checkpoint_path=ckpt_path,
            model_name=cfg.model_name,
            dataloader=loader,
            device=cfg.device,
        )
        elapsed = time.time() - t0
        print(f"       {len(patch_scores)} patches scored in {elapsed:.1f}s")

        all_patch_scores[fold_key] = patch_scores
        all_stack_scores[fold_key] = aggregate_patches_to_stacks(
            patch_scores, patch_to_stack
        )

    # ── 4. Compute per-model metrics ──────────────────────────────
    print(f"[4/6] Computing per-model metrics...")
    per_model_metrics = {}
    for fold_key, stacks in all_stack_scores.items():
        per_model_metrics[fold_key] = compute_metrics(stacks, stack_labels)

    # ── 5. Ensemble ───────────────────────────────────────────────
    print(f"[5/6] Ensembling {cfg.n_folds} models...")
    ensemble_scores = ensemble_stacks(list(all_stack_scores.values()))
    ensemble_metrics = compute_metrics(ensemble_scores, stack_labels)

    print(f"       Ensemble F1 mean: {ensemble_metrics['f1_mean']}")
    print(f"       Ensemble AUC:     {ensemble_metrics['auc']}")

    # ── 6. Save report ────────────────────────────────────────────
    results = {
        "config": {
            "model_name": cfg.model_name,
            "n_folds": cfg.n_folds,
            "preprocessed_dir": cfg.preprocessed_dir,
            "cv_global_json": cfg.cv_global_json,
            "batch_size": cfg.batch_size,
            "checkpoints": cfg.checkpoints,
        },
        "patch_scores": all_patch_scores,
        "stack_scores": {
            "per_model": all_stack_scores,
            "ensemble": ensemble_scores,
        },
        "labels": stack_labels,
        "predictions": {
            sid: int(score >= 0.5)
            for sid, score in ensemble_scores.items()
        },
        "metrics": {
            "per_model": per_model_metrics,
            "ensemble": ensemble_metrics,
        },
    }

    print(f"[6/6] Saving results to {cfg.results_dir}...")
    json_path = save_results_json(cfg.results_dir, results)
    txt_path = save_summary_txt(cfg.results_dir, results)

    print(f"       {json_path}")
    print(f"       {txt_path}")
    print("Done!")


if __name__ == "__main__":
    main()
