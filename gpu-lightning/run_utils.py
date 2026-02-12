"""Run directory setup and metadata copy utilities."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def create_run_directories(runs_root: Path, run_name: str) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{run_name}_{timestamp}"

    run_dir = runs_root / unique_run_name
    dirs = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "results": run_dir / "results",
        "analytics": run_dir / "analytics",
        "wandb": run_dir / "wandb",
        "data": run_dir / "data",
        "run_name": unique_run_name,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    for key, path in dirs.items():
        if key == "run_name":
            continue
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_run_metadata(
    preprocessed_dir: Path,
    split_json: Path,
    data_dir: Path,
    config_path: Path,
    dataset_json: Optional[Path] = None,
) -> None:
    _copy_if_exists(split_json, data_dir / "train_test_split.json")
    _copy_if_exists(preprocessed_dir / "metadata.json", data_dir / "preprocessing_metadata.json")
    _copy_if_exists(preprocessed_dir / "patches_info.json", data_dir / "patches_info.json")
    _copy_if_exists(config_path, data_dir / "config.yaml")

    if dataset_json is not None:
        _copy_if_exists(dataset_json, data_dir / "original_dataset.json")
    else:
        meta = preprocessed_dir / "metadata.json"
        if meta.exists():
            try:
                metadata = json.loads(meta.read_text(encoding="utf-8"))
                src = metadata.get("dataset_source")
                if src:
                    _copy_if_exists(Path(src), data_dir / "original_dataset.json")
            except Exception:
                pass


def save_wandb_info(data_dir: Path, project: str, run_name: str, run_id: Optional[str], url: Optional[str]) -> None:
    info = {
        "project": project,
        "run_name": run_name,
        "run_id": run_id,
        "url": url,
    }
    out = data_dir / "wandb_info.json"
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_training_summary(results_dir: Path, best_model_path: str, best_val_f1_mean: float) -> None:
    summary = {
        "best_model_path": best_model_path,
        "best_val_f1_mean": best_val_f1_mean,
    }
    out = results_dir / "training_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def move_best_checkpoint_to_pth(checkpoints_dir: Path, best_ckpt_path: str) -> Path:
    src = Path(best_ckpt_path)
    if not src.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")
    dst = checkpoints_dir / "best_model.pth"
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    for extra in checkpoints_dir.glob("*.ckpt"):
        if extra != dst:
            extra.unlink(missing_ok=True)
    return dst


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
