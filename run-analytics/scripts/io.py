"""I/O utilities."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union


def resolve_run_paths(args: List[str], print_skips: bool = False) -> List[str]:
    """Expand globs and return paths that have results/training_results.json."""
    paths = []
    for arg in args:
        p = Path(arg)
        if p.exists():
            paths.append(str(p))
        else:
            for m in p.parent.glob(p.name):
                paths.append(str(m))
    valid = []
    for p in paths:
        if (Path(p) / "results" / "training_results.json").exists():
            valid.append(p)
        elif print_skips:
            print(f"  Skip (no results): {p}")
    return valid


def load_results(path: Union[Path, str]) -> Dict:
    """Load training_results.json from a run dir or direct path."""
    p = Path(path).resolve()
    if p.is_dir():
        p = p / "results" / "training_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Results not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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
