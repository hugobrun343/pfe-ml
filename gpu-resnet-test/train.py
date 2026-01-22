#!/usr/bin/env python3
"""Training script for ResNet3D-50 binary classification"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import yaml

from model import create_model
from data_loader import create_dataloader
from trainer import Trainer
from results import ResultsTracker
from wandb_utils import init_wandb_from_args, finalize_wandb


def load_train_val_data(args):
    """
    Load training and validation data loaders from .nii.gz patches
    
    Args:
        args: Parsed command line arguments (must have preprocessed_dir and train_test_split_json)
        
    Returns:
        Tuple of (train_loader, val_loader, val_sampler). val_loader may be None
    """
    preprocessed_dir = Path(args.preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {args.preprocessed_dir}")
    
    train_test_split_json = Path(args.train_test_split_json)
    if not train_test_split_json.exists():
        raise FileNotFoundError(f"Train/test split file not found: {args.train_test_split_json}")
    
    # Build paths
    patches_dir = preprocessed_dir / "patches"
    patches_info_json = preprocessed_dir / "patches_info.json"
    
    if not patches_dir.exists():
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")
    if not patches_info_json.exists():
        raise FileNotFoundError(f"Patches info JSON not found: {patches_info_json}")
    
    print(f"\nLoading data from: {preprocessed_dir}")
    print(f"  Train/test split: {train_test_split_json}")
    
    # Create training data loader
    print(f"  Loading training data...")
    train_loader, _ = create_dataloader(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        track_indices=False
    )
    
    # Create validation data loader
    print(f"  Loading validation data...")
    val_loader, val_sampler = create_dataloader(
        patches_dir=str(patches_dir),
        patches_info_json=str(patches_info_json),
        train_test_split_json=str(train_test_split_json),
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        track_indices=True  # Track indices for validation to get stack_ids and grid_positions
    )
    
    return train_loader, val_loader, val_sampler


def setup_model_and_optimizer(args, device):
    """
    Create model, loss function and optimizer
    
    Args:
        args: Parsed command line arguments
        device: Device to use (cuda/cpu)
        
    Returns:
        Tuple of (model, criterion, optimizer)
    """
    model = create_model(in_channels=args.in_channels, num_classes=1)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    return model, criterion, optimizer


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args(config: dict):
    """
    Parse command line arguments, using config values
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        Parsed arguments namespace with CLI overrides applied
    """
    parser = argparse.ArgumentParser(description='Train ResNet3D for binary classification')
    
    # Extract config sections
    training = config.get('training', {})
    model = config.get('model', {})
    input_cfg = config.get('input', {})
    system = config.get('system', {})
    output = config.get('output', {})
    data_cfg = config.get('data', {})
    
    # Data paths (from YAML only, CLI can override)
    default_preprocessed_dir = data_cfg.get('preprocessed_dir')
    default_train_test_split_json = data_cfg.get('train_test_split_json')
    parser.add_argument('--preprocessed-dir', type=str, default=default_preprocessed_dir,
                       help='Path to preprocessed directory (contains patches/ and patches_info.json)')
    parser.add_argument('--train-test-split-json', type=str, default=default_train_test_split_json,
                       help='Path to train_test_split.json file')
    
    # Training parameters (from YAML only, CLI can override)
    parser.add_argument('--batch-size', type=int, default=input_cfg.get('batch_size'),
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=training.get('epochs'),
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=training.get('learning_rate'),
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=training.get('weight_decay'),
                       help='Weight decay')
    parser.add_argument('--in-channels', type=int, default=model.get('in_channels'),
                       help='Number of input channels')
    
    parser.add_argument('--device', type=str, default=system.get('device'),
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=training.get('num_workers'),
                       help='Number of data loader workers')
    parser.add_argument('--prefetch-factor', type=int, default=training.get('prefetch_factor'),
                       help='Number of batches each worker prefetches in advance')
    parser.add_argument('--save-dir', type=str, default=output.get('save_dir'),
                       help='Directory to save checkpoints')
    
    parser.add_argument('--early-stopping-patience', type=int,
                       default=training.get('early_stopping_patience'),
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--early-stopping-min-delta', type=float,
                       default=training.get('early_stopping_min_delta'),
                       help='Minimum change to qualify as improvement')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., checkpoints/.../latest_model.pth)')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Validate that required values are present (from YAML or CLI)
    if args.preprocessed_dir is None:
        raise ValueError("Preprocessed directory must be specified in config.yaml (data.preprocessed_dir) or via --preprocessed-dir")
    if args.train_test_split_json is None:
        raise ValueError("Train/test split JSON must be specified in config.yaml (data.train_test_split_json) or via --train-test-split-json")
    
    return args


def main():
    """Main training function"""
    # Parse CLI args first to get config path
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument('--config', type=str, required=True,
                               help='Path to configuration YAML file')
    initial_args, _ = initial_parser.parse_known_args()
    
    # Load config from YAML
    config = load_config(initial_args.config)
    print(f"Configuration loaded from: {initial_args.config}")
    
    # Parse full args with config as defaults (CLI can override)
    args = parse_args(config)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nResNet3D-50 Training")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"  Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    
    train_loader, val_loader, val_sampler = load_train_val_data(args)
    model, criterion, optimizer = setup_model_and_optimizer(args, device)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 1
    resume_mode = args.resume is not None
    if resume_mode:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1_class_1 = checkpoint.get('val_f1_class_1', checkpoint.get('val_f1', 0.0))
        print(f"  Resumed from epoch {checkpoint['epoch']}, continuing from epoch {start_epoch}")
    else:
        best_val_f1_class_1 = 0.0
    
    # Prepare config dict for ResultsTracker
    results_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        'in_channels': args.in_channels,
        'device': str(device),
    }
    
    # Initialize wandb
    args.device = device  # Add device to args for wandb
    init_wandb_from_args(args, model, save_dir)
    
    results_tracker = ResultsTracker(results_config, save_dir, resume=resume_mode)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        results_tracker=results_tracker,
        val_sampler=val_sampler,
        best_val_f1_class_1=best_val_f1_class_1
    )
    
    summary = trainer.train(args.epochs, start_epoch=start_epoch)
    
    json_path = results_tracker.finalize(
        best_val_f1_class_1=summary['best_val_f1_class_1'] if val_loader else None,
        total_epochs=summary['total_epochs'],
        early_stopped=summary['early_stopped']
    )
    
    # Finalize wandb
    finalize_wandb(summary, has_validation=val_loader is not None)
    
    print("\nTraining completed!")
    if val_loader:
        # Display final confusion matrix and F1 scores per class
        if summary.get('final_val_metrics'):
            final_metrics = summary['final_val_metrics']
            tp = final_metrics.get('tp', 0)
            fp = final_metrics.get('fp', 0)
            tn = final_metrics.get('tn', 0)
            fn = final_metrics.get('fn', 0)
            f1_class_0 = final_metrics.get('f1_class_0', 0.0)
            f1_class_1 = final_metrics.get('f1_class_1', 0.0)
            
            print("\n" + "="*50)
            print("FINAL VALIDATION RESULTS")
            print("="*50)
            print(f"\nConfusion Matrix:")
            print(f"  {'':<15} {'Predicted 0':<15} {'Predicted 1':<15}")
            print(f"  {'Actual 0':<15} {tn:<15} {fp:<15}")
            print(f"  {'Actual 1':<15} {fn:<15} {tp:<15}")
            print(f"\nF1 Scores:")
            print(f"  Class 0 (Negative): {f1_class_0:.4f}")
            print(f"  Class 1 (Positive): {f1_class_1:.4f}")
            print("="*50)
            
            # Save confusion matrix to file
            confusion_matrix_path = save_dir / 'confusion_matrix.txt'
            with open(confusion_matrix_path, 'w') as f:
                f.write("="*50 + "\n")
                f.write("FINAL VALIDATION RESULTS\n")
                f.write("="*50 + "\n")
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"  {'':<15} {'Predicted 0':<15} {'Predicted 1':<15}\n")
                f.write(f"  {'Actual 0':<15} {tn:<15} {fp:<15}\n")
                f.write(f"  {'Actual 1':<15} {fn:<15} {tp:<15}\n")
                f.write(f"\nF1 Scores:\n")
                f.write(f"  Class 0 (Negative): {f1_class_0:.4f}\n")
                f.write(f"  Class 1 (Positive): {f1_class_1:.4f}\n")
                f.write("="*50 + "\n")
            print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
    else:
        print("No validation set used")
    print(f"\nDetailed results saved to: {json_path}")


if __name__ == "__main__":
    print("Starting training...")
    main()
