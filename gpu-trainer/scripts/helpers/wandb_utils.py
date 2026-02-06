"""Weights & Biases logging utilities"""

import wandb
from pathlib import Path
from typing import Dict, Any, Optional
from argparse import Namespace


def init_wandb_from_args(args: Namespace, model, save_dir: Path, project_name: str = 'resnet3d-binary-classification', run_name: str = None, group: str = None, resume_id: Optional[str] = None, config_path: Optional[Path] = None, preprocessing_metadata_path: Optional[Path] = None):
    """
    Initialize Weights & Biases logging from argparse args.
    
    If resume_id is provided, resumes the existing wandb run.
    Otherwise, creates a new run with the given run_name.
    
    Args:
        args: Parsed command line arguments
        model: Model to watch
        save_dir: Directory to save wandb logs
        project_name: W&B project name
        run_name: Name for the W&B run (required if resume_id is None)
        resume_id: W&B run ID to resume (if None, creates new run)
        config_path: Path to config.yaml file to upload to wandb
        preprocessing_metadata_path: Path to preprocessing_metadata.json to upload to wandb
    """
    # Use provided run_name or fallback to save_dir name
    if run_name is None:
        run_name = save_dir.name if hasattr(save_dir, 'name') else str(save_dir).split('/')[-1]
    
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'in_channels': args.in_channels,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        'device': str(args.device) if hasattr(args, 'device') else 'cuda',
        'preprocessed_dir': str(args.preprocessed_dir) if hasattr(args, 'preprocessed_dir') else None,
        'train_test_split_json': str(args.train_test_split_json) if hasattr(args, 'train_test_split_json') else None,
    }
    
    # Resume existing run or create new one
    if resume_id:
        print(f"  Resuming wandb run: {resume_id}")
        wandb.init(
            project=project_name,
            id=resume_id,
            resume="allow",
            name=run_name,
            group=group,
            config=config,
            dir=str(save_dir)
        )
    else:
        print(f"  Creating new wandb run: {run_name}" + (f" (group: {group})" if group else ""))
        wandb.init(
            project=project_name,
            name=run_name,
            group=group,
            config=config,
            dir=str(save_dir)
        )
    
    # Upload config.yaml file to wandb if provided
    if config_path and config_path.exists():
        wandb.save(str(config_path), base_path=config_path.parent, policy='now')
        print(f"  Uploaded config file to wandb: {config_path.name}")
    elif config_path:
        print(f"  Warning: Config file not found at {config_path}")
    
    # Upload preprocessing metadata to wandb if provided
    if preprocessing_metadata_path and preprocessing_metadata_path.exists():
        wandb.save(str(preprocessing_metadata_path), base_path=preprocessing_metadata_path.parent, policy='now')
        print(f"  Uploaded preprocessing metadata to wandb: {preprocessing_metadata_path.name}")
    elif preprocessing_metadata_path:
        print(f"  Warning: Preprocessing metadata not found at {preprocessing_metadata_path}")
    
    wandb.watch(model, log='gradients', log_freq=100)
    
    # Return the run ID and URL for saving
    return wandb.run.id, wandb.run.url


def log_epoch_metrics(epoch: int, epoch_time: float, train_metrics: Dict[str, float], 
                     val_metrics: Optional[Dict[str, Any]] = None):
    """
    Log epoch metrics to wandb
    
    Args:
        epoch: Current epoch number
        epoch_time: Time taken for epoch (seconds)
        train_metrics: Dict with 'loss', 'f1_class_0', 'f1_class_1', 'f1_macro'
        val_metrics: Dict with 'loss', 'f1_class_0', 'f1_class_1', 'f1_macro', etc. (optional)
    """
    log_dict = {
        'epoch': epoch,
        'train/loss': train_metrics['loss'],
        'train/f1_class_0': train_metrics.get('f1_class_0', 0.0),
        'train/f1_class_1': train_metrics.get('f1_class_1', 0.0),
        'train/f1_macro': train_metrics.get('f1_macro', 0.0),
        'epoch_time': epoch_time
    }
    
    if val_metrics:
        log_dict.update({
            'val/loss': val_metrics['loss'],
            'val/f1_class_0': val_metrics.get('f1_class_0', 0.0),
            'val/f1_class_1': val_metrics.get('f1_class_1', 0.0),
            'val/f1_macro': val_metrics.get('f1_macro', 0.0),
            'val/accuracy': val_metrics['accuracy'],
            'val/precision': val_metrics['precision'],
            'val/recall': val_metrics['recall'],
            'val/tp': val_metrics['tp'],
            'val/fp': val_metrics['fp'],
            'val/tn': val_metrics['tn'],
            'val/fn': val_metrics['fn'],
        })
    
    wandb.log(log_dict, step=epoch)


def finalize_wandb(summary: Dict[str, Any], has_validation: bool = True):
    """
    Finalize wandb run with summary metrics
    
    Args:
        summary: Training summary dict with 'best_val_f1_macro', 'total_epochs', 'early_stopped'
        has_validation: Whether validation set was used
    """
    if has_validation and 'best_val_f1_macro' in summary:
        wandb.run.summary['best_val_f1_macro'] = summary['best_val_f1_macro']
    wandb.run.summary['total_epochs'] = summary['total_epochs']
    wandb.run.summary['early_stopped'] = summary['early_stopped']
    wandb.finish()
