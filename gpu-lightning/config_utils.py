"""Configuration loading for gpu-lightning training."""

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (dictionary at root).")
    return cfg


def parse_train_config(config: Dict[str, Any], config_path: str, project_root: Path) -> Namespace:
    run = config.get("run", {})
    data = config.get("data", {})
    input_cfg = config.get("input", {})
    training = config.get("training", {})
    model = config.get("model", {})
    wandb_cfg = config.get("wandb", {})
    system = config.get("system", {})
    early = training.get("early_stopping", {}) if isinstance(training.get("early_stopping", {}), dict) else {}

    args = Namespace()
    args.config = str(Path(config_path).resolve())

    args.run_name = run.get("run_name")
    args.runs_root = run.get("runs_root", str((project_root.parent / "_runs").resolve()))

    args.preprocessed_dir = data.get("preprocessed_dir")
    args.train_test_split_json = data.get("train_test_split_json")
    args.dataset_json = data.get("dataset_json")

    args.batch_size = input_cfg.get("batch_size", 32)

    args.epochs = training.get("epochs", 10)
    args.lr = training.get("learning_rate", 1e-3)
    args.limit_train_batches = training.get("limit_train_batches", 1.0)

    # Optimizer / scheduler (optional, backward-compatible defaults)
    args.optimizer = training.get("optimizer", "adam")
    args.weight_decay = training.get("weight_decay", 0.0)
    args.warmup_epochs = training.get("warmup_epochs", 0)
    args.scheduler = training.get("scheduler", None)
    args.gradient_clip_val = training.get("gradient_clip_val", None)

    args.early_stopping_enabled = early.get("enabled", False)
    args.early_stopping_patience = early.get("patience", 3)
    args.early_stopping_min_delta = early.get("min_delta", 0.0)

    args.model_name = model.get("name", "resnet3d_50")

    args.wandb_project = wandb_cfg.get("project", "gpu-lightning")
    args.wandb_run_name = wandb_cfg.get("run_name")
    args.wandb_group = wandb_cfg.get("group")

    args.accelerator = system.get("accelerator")
    args.devices = system.get("devices")
    args.precision = system.get("precision")

    _validate_required(args)
    return args


def _validate_required(args: Namespace) -> None:
    required = {
        "run.run_name": args.run_name,
        "data.preprocessed_dir": args.preprocessed_dir,
        "data.train_test_split_json": args.train_test_split_json,
    }
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
