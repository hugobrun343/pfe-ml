"""Rank stacks by how problematic they are across multiple runs."""

from collections import defaultdict
from typing import Dict, List, Tuple

from .io import load_results


def _extract_volumes(results: dict) -> List[Dict]:
    """Extract volume-level results from last epoch."""
    epochs = results.get("epochs", [])
    if not epochs:
        return []
    val = epochs[-1].get("validation", {})
    return val.get("volumes", [])


def rank_problematic_stacks(
    results_paths: List[str],
    top_n: int = 20,
) -> List[Tuple[str, float, Dict]]:
    """
    Rank stacks by how often they are misclassified across runs.
    Score = fraction of runs where the stack was wrong (higher = more problematic).

    Args:
        results_paths: Paths to run dirs or training_results.json files.
        top_n: Number of worst stacks to return.

    Returns:
        List of (stack_id, score, details) sorted by score descending.
    """
    stack_data = defaultdict(lambda: {"correct": 0, "total": 0, "probs": []})

    for path in results_paths:
        results = load_results(path)
        volumes = _extract_volumes(results)
        for v in volumes:
            sid = v.get("stack_id")
            if not sid:
                continue
            stack_data[sid]["total"] += 1
            if v.get("correct", False):
                stack_data[sid]["correct"] += 1
            stack_data[sid]["probs"].append(
                (v.get("aggregated_probability"), v.get("true_label"))
            )

    ranked = []
    for sid, data in stack_data.items():
        total = data["total"]
        correct = data["correct"]
        wrong = total - correct
        score = wrong / total if total > 0 else 0.0
        ranked.append((
            sid,
            score,
            {"wrong": wrong, "total": total, "correct": correct},
        ))

    ranked.sort(key=lambda x: (-x[1], -x[2]["wrong"]))
    return ranked[:top_n]
