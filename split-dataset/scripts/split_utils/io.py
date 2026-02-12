"""Load and save helpers."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict:
    """Load a YAML configuration file."""
    print(f"Loading config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: Path) -> Dict:
    """Load the enriched dataset JSON."""
    print(f"Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data.get('stacks', []))} stacks")
    return data


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write *data* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    print(f"  Saved: {path}")
