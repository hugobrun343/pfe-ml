"""Step 4: Train the model with validation (config-driven)."""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config_utils import load_config, parse_train_config
from run_utils import (
    copy_run_metadata,
    create_run_directories,
    move_best_checkpoint_to_pth,
    save_wandb_info,
    write_training_summary,
)
from step2_lightning_module import Lit3DClassifier
from step3_dataset import make_train_val_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning step 4: train + val")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt_path", default=None, help="Path to .ckpt to resume training from")
    parser.add_argument("--wandb_resume_id", default=None, help="WandB run ID to resume (shows as continued run)")
    parser.add_argument("--run_dir", default=None, help="Existing run directory to reuse (for resume)")
    return parser.parse_args()


def main():
    cli = parse_args()
    project_root = Path(__file__).parent

    raw_cfg = load_config(cli.config)
    cfg = parse_train_config(raw_cfg, cli.config, project_root)

    if cli.run_dir:
        # Resume mode: reuse existing run directory
        run_dir = Path(cli.run_dir)
        run_dirs = {
            "run_dir": run_dir,
            "checkpoints": run_dir / "checkpoints",
            "results": run_dir / "results",
            "analytics": run_dir / "analytics",
            "wandb": run_dir / "wandb",
            "data": run_dir / "data",
            "run_name": run_dir.name,
        }
        for key, path in run_dirs.items():
            if key != "run_name" and isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
        run_name = run_dirs["run_name"]
    else:
        run_dirs = create_run_directories(Path(cfg.runs_root), cfg.run_name)
        run_name = run_dirs["run_name"]

        copy_run_metadata(
            preprocessed_dir=Path(cfg.preprocessed_dir),
            split_json=Path(cfg.train_test_split_json),
            data_dir=run_dirs["data"],
            config_path=Path(cfg.config),
            dataset_json=Path(cfg.dataset_json) if cfg.dataset_json else None,
        )

    train_loader, val_loader = make_train_val_loaders(
        preprocessed_dir=cfg.preprocessed_dir,
        splits_json=cfg.train_test_split_json,
        batch_size=cfg.batch_size,
    )

    model = Lit3DClassifier(
        model_name=cfg.model_name,
        lr=cfg.lr,
        optimizer=cfg.optimizer,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        scheduler=cfg.scheduler,
        max_epochs=cfg.epochs,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dirs["checkpoints"]),
        filename="best_model",
        monitor="val_f1_mean",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_cb]
    if cfg.early_stopping_enabled:
        callbacks.append(
            EarlyStopping(
                monitor="val_f1_mean",
                mode="max",
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                verbose=True,
            )
        )

    wandb_run_name = cfg.wandb_run_name or run_name
    wandb_kwargs = {
        "project": cfg.wandb_project,
        "name": wandb_run_name,
        "group": cfg.wandb_group,
        "save_dir": str(run_dirs["wandb"]),
        "log_model": False,
    }
    if cli.wandb_resume_id:
        wandb_kwargs["id"] = cli.wandb_resume_id
        wandb_kwargs["resume"] = "must"
    wandb_logger = WandbLogger(**wandb_kwargs)

    trainer_kwargs = {
        "default_root_dir": str(run_dirs["run_dir"]),
        "logger": wandb_logger,
        "callbacks": callbacks,
        "max_epochs": cfg.epochs,
        "limit_train_batches": cfg.limit_train_batches,
    }
    if cfg.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = cfg.gradient_clip_val
    if cfg.accelerator is not None:
        trainer_kwargs["accelerator"] = cfg.accelerator
    if cfg.devices is not None:
        trainer_kwargs["devices"] = cfg.devices
    if cfg.precision is not None:
        trainer_kwargs["precision"] = cfg.precision

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cli.ckpt_path,
    )

    best_ckpt_path = checkpoint_cb.best_model_path
    if not best_ckpt_path:
        raise RuntimeError("No best checkpoint was produced. Check validation metrics logging.")

    best_model_path = move_best_checkpoint_to_pth(run_dirs["checkpoints"], best_ckpt_path)
    best_score = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score is not None else None
    write_training_summary(run_dirs["results"], str(best_model_path), float(best_score) if best_score is not None else -1.0)

    run_id = getattr(wandb_logger.experiment, "id", None) if wandb_logger.experiment is not None else None
    run_url = getattr(wandb_logger.experiment, "url", None) if wandb_logger.experiment is not None else None
    save_wandb_info(run_dirs["data"], cfg.wandb_project, wandb_run_name, run_id, run_url)

    print(f"Run directory: {run_dirs['run_dir']}")
    print(f"Best model: {best_model_path}")
    if best_score is not None:
        print(f"Best val_f1_mean: {best_score:.6f}")


if __name__ == "__main__":
    main()
