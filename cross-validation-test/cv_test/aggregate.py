"""Aggregate patch scores to stack scores and ensemble multiple models."""

from collections import defaultdict
from typing import Dict, List, Tuple


def aggregate_patches_to_stacks(
    patch_scores: Dict[str, float],
    patch_to_stack: Dict[str, str],
) -> Dict[str, float]:
    """Aggregate per-patch probabilities into per-stack scores (mean).

    Args:
        patch_scores: {patch_filename: probability}
        patch_to_stack: {patch_filename: stack_id}

    Returns:
        {stack_id: mean_probability}
    """
    stack_accum: Dict[str, List[float]] = defaultdict(list)

    for patch_file, prob in patch_scores.items():
        stack_id = patch_to_stack[patch_file]
        stack_accum[stack_id].append(prob)

    return {
        stack_id: sum(probs) / len(probs)
        for stack_id, probs in stack_accum.items()
    }


def ensemble_stacks(
    per_model_stacks: List[Dict[str, float]],
) -> Dict[str, float]:
    """Ensemble multiple models by averaging their stack-level scores.

    Args:
        per_model_stacks: List of dicts, one per model fold,
                          each mapping {stack_id: mean_probability}.

    Returns:
        {stack_id: mean_probability_across_models}
    """
    stack_accum: Dict[str, List[float]] = defaultdict(list)

    for model_scores in per_model_stacks:
        for stack_id, score in model_scores.items():
            stack_accum[stack_id].append(score)

    return {
        stack_id: sum(scores) / len(scores)
        for stack_id, scores in stack_accum.items()
    }
