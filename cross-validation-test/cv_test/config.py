"""Parse YAML configuration for CV test pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class CVTestConfig:
    """All settings for a cross-validation test run."""

    # Model
    model_name: str

    # Paths
    preprocessed_dir: str
    cv_global_json: str

    # Checkpoints (one per fold)
    checkpoints: List[str]

    # Inference
    batch_size: int = 32
    device: str = "cuda"

    # Output
    results_dir: str = "_results/cv_test"

    @property
    def n_folds(self) -> int:
        return len(self.checkpoints)

    def validate(self) -> None:
        """Check that all referenced files/dirs exist."""
        errors = []

        if not Path(self.preprocessed_dir).is_dir():
            errors.append(f"preprocessed_dir not found: {self.preprocessed_dir}")

        if not Path(self.cv_global_json).is_file():
            errors.append(f"cv_global_json not found: {self.cv_global_json}")

        for i, ckpt in enumerate(self.checkpoints):
            if not Path(ckpt).is_file():
                errors.append(f"checkpoint fold {i} not found: {ckpt}")

        if len(self.checkpoints) == 0:
            errors.append("No checkpoints provided")

        if errors:
            raise FileNotFoundError(
                "Config validation failed:\n  " + "\n  ".join(errors)
            )


def load_config(config_path: str) -> CVTestConfig:
    """Load and parse a YAML config file into a CVTestConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping at root level.")

    model_cfg = raw.get("model", {})
    paths_cfg = raw.get("paths", {})
    inference_cfg = raw.get("inference", {})
    output_cfg = raw.get("output", {})
    checkpoints = raw.get("checkpoints", [])

    return CVTestConfig(
        model_name=model_cfg["name"],
        preprocessed_dir=paths_cfg["preprocessed_dir"],
        cv_global_json=paths_cfg["cv_global_json"],
        checkpoints=checkpoints,
        batch_size=inference_cfg.get("batch_size", 32),
        device=inference_cfg.get("device", "cuda"),
        results_dir=output_cfg.get("results_dir", "_results/cv_test"),
    )
