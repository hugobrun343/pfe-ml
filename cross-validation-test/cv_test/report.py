"""Save results (JSON + human-readable summary)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_results_json(results_dir: str, results: Dict[str, Any]) -> Path:
    """Write full results dict to results.json."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


def save_summary_txt(results_dir: str, results: Dict[str, Any]) -> Path:
    """Write a human-readable summary.txt."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "summary.txt"
    lines = _build_summary(results)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path


def _build_summary(results: Dict[str, Any]) -> list:
    """Build the text lines for summary.txt."""
    cfg = results.get("config", {})
    ens = results.get("metrics", {}).get("ensemble", {})
    per_model = results.get("metrics", {}).get("per_model", {})

    lines = [
        "=" * 60,
        "  CROSS-VALIDATION TEST REPORT",
        "=" * 60,
        "",
        f"Model:       {cfg.get('model_name', '?')}",
        f"Folds:       {cfg.get('n_folds', '?')}",
        f"Test stacks: {ens.get('n_samples', '?')}",
        f"Threshold:   {ens.get('threshold', 0.5)}",
        f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 60,
        "  ENSEMBLE METRICS (mean of 5 fold models)",
        "-" * 60,
        "",
        f"  F1 positive:   {ens.get('f1_pos', '?')}",
        f"  F1 negative:   {ens.get('f1_neg', '?')}",
        f"  F1 mean:       {ens.get('f1_mean', '?')}",
        f"  Accuracy:      {ens.get('accuracy', '?')}",
        f"  AUC:           {ens.get('auc', '?')}",
        "",
    ]

    # Confusion matrix
    cm = ens.get("confusion_matrix", {})
    if cm:
        lines += [
            "  Confusion Matrix:",
            f"                  Predicted 0    Predicted 1",
            f"    Actual 0       {cm.get('tn', '?'):>6}         {cm.get('fp', '?'):>6}",
            f"    Actual 1       {cm.get('fn', '?'):>6}         {cm.get('tp', '?'):>6}",
            "",
        ]

    # Per-model metrics
    if per_model:
        lines += [
            "-" * 60,
            "  PER-MODEL (fold) METRICS",
            "-" * 60,
            "",
        ]
        for fold_key in sorted(per_model.keys()):
            m = per_model[fold_key]
            lines.append(
                f"  {fold_key:12s}  "
                f"F1m={m.get('f1_mean', '?'):.4f}  "
                f"Acc={m.get('accuracy', '?'):.4f}  "
                f"AUC={m.get('auc', '?'):.4f}"
            )
        lines.append("")

    # Per-stack details (ensemble)
    per_stack = ens.get("per_stack", [])
    if per_stack:
        lines += [
            "-" * 60,
            "  PER-STACK DETAILS (ensemble)",
            "-" * 60,
            "",
            f"  {'Stack ID':>16s}  Label  Score   Pred  Correct",
            f"  {'-'*16:>16s}  -----  ------  ----  -------",
        ]
        for s in per_stack:
            mark = "OK" if s["correct"] else "MISS"
            lines.append(
                f"  {s['stack_id']:>16s}  "
                f"{s['label']:>5d}  "
                f"{s['score']:>6.4f}  "
                f"{s['prediction']:>4d}  "
                f"{mark:>7s}"
            )
        lines.append("")

    lines.append("=" * 60)
    return lines
