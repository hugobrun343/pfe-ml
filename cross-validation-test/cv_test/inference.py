"""Run inference with a single trained model on the test holdout set."""

from typing import Dict

import torch
from torch.utils.data import DataLoader


def run_inference(
    checkpoint_path: str,
    model_name: str,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run a single fold model on the test set.

    Args:
        checkpoint_path: Path to the best_model.pth checkpoint.
        model_name: Model name (must match MODEL_REGISTRY key).
        dataloader: Test holdout DataLoader.
        device: Device string ("cuda" or "cpu").

    Returns:
        Dict mapping patch_filename -> sigmoid probability.
    """
    from lightning_module import Lit3DClassifier

    # Load model from checkpoint
    model = Lit3DClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()

    patch_scores: Dict[str, float] = {}

    with torch.no_grad():
        for batch in dataloader:
            tensors, labels, filenames, stack_ids = batch
            tensors = tensors.to(device)

            logits = model(tensors).view(-1)
            probs = torch.sigmoid(logits).cpu().tolist()

            for fname, prob in zip(filenames, probs):
                patch_scores[fname] = prob

    return patch_scores
