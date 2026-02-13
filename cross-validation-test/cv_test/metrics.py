"""Compute classification metrics from stack-level scores."""

from typing import Any, Dict, List

import numpy as np


def compute_metrics(
    stack_scores: Dict[str, float],
    stack_labels: Dict[str, int],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute F1 (pos/neg/mean), accuracy, AUC, and confusion matrix.

    Args:
        stack_scores: {stack_id: probability}
        stack_labels: {stack_id: ground_truth_label (0 or 1)}
        threshold: Classification threshold.

    Returns:
        Dict with all metrics + per-stack details.
    """
    # Align stacks
    stack_ids = sorted(stack_scores.keys())
    y_true = np.array([stack_labels[sid] for sid in stack_ids])
    y_prob = np.array([stack_scores[sid] for sid in stack_ids])
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix components
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    eps = 1e-8

    # F1 scores
    f1_pos = (2 * tp) / (2 * tp + fp + fn + eps)
    f1_neg = (2 * tn) / (2 * tn + fn + fp + eps)
    f1_mean = (f1_pos + f1_neg) / 2.0

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    # AUC (manual implementation, no sklearn dependency)
    auc = _compute_auc(y_true, y_prob)

    # Per-stack details
    per_stack = [
        {
            "stack_id": sid,
            "label": int(stack_labels[sid]),
            "score": round(stack_scores[sid], 5),
            "prediction": int(y_prob[i] >= threshold),
            "correct": bool(y_pred[i] == y_true[i]),
        }
        for i, sid in enumerate(stack_ids)
    ]

    return {
        "f1_pos": round(f1_pos, 5),
        "f1_neg": round(f1_neg, 5),
        "f1_mean": round(f1_mean, 5),
        "accuracy": round(accuracy, 5),
        "auc": round(auc, 5),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_samples": len(stack_ids),
        "threshold": threshold,
        "per_stack": per_stack,
    }


def _compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC-ROC using the trapezoidal rule (no sklearn needed).

    Handles edge cases (single class, ties).
    """
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by descending score
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    # Accumulate TPR and FPR
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg

    # Prepend origin (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration
    auc = float(np.trapz(tpr, fpr))
    return auc
