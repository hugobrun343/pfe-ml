"""I/O utilities."""

import json
from pathlib import Path
from typing import Dict, Tuple


def load_run(run_path: Path) -> Tuple[Dict, Dict]:
    """Load training_results.json and train_test_split.json."""
    run_path = Path(run_path).resolve()
    results_path = run_path / "results" / "training_results.json"
    split_path = run_path / "data" / "train_test_split.json"

    if not results_path.exists():
        raise FileNotFoundError(f"training_results.json not found: {results_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"train_test_split.json not found: {split_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    return results, split
