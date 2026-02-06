#!/usr/bin/env python3
"""Training script for ResNet3D-50 binary classification"""

import torch
from pathlib import Path
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))
sys.path.insert(0, str(scripts_dir / "helpers"))

# Local imports
from scripts.helpers.config_loader import load_config, parse_args
from scripts.helpers.run_setup import create_run_directories, copy_data_metadata, copy_config, get_runs_directory
from scripts.helpers.data_loading import load_data
from scripts.helpers.model_setup import setup_model_and_optimizer
from scripts.helpers.resume import resume_from_checkpoint
from scripts.helpers.display import print_training_info, print_final_results
from scripts.helpers.trainer import Trainer
from scripts.helpers.results import ResultsTracker
from scripts.helpers.wandb_utils import init_wandb_from_args, finalize_wandb


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
    
    # Get runs directory (points to _runs at work-hugo root)
    runs_dir = get_runs_directory(project_root)
    
    # If resuming, extract run name from resume path (use existing folder name)
    # Otherwise, create new run with timestamp
    resume_mode = args.resume is not None
    if resume_mode:
        # Extract run name from resume path (e.g., _runs/my_run_20260122_143052/checkpoints/latest_model.pth -> my_run_20260122_143052)
        resume_path = Path(args.resume)
        # Go up to find the run directory
        if resume_path.name in ['latest_model.pth', 'best_model.pth']:
            run_dir_from_resume = resume_path.parent.parent
        else:
            run_dir_from_resume = resume_path.parent
        actual_run_name = run_dir_from_resume.name
        # Use existing directory structure (don't create new one with timestamp)
        run_dir = runs_dir / actual_run_name
        run_dirs = {
            'run_dir': run_dir,
            'checkpoints': run_dir / 'checkpoints',
            'results': run_dir / 'results',
            'analytics': run_dir / 'analytics',
            'wandb': run_dir / 'wandb',
            'data': run_dir / 'data',
            'run_name': actual_run_name
        }
    else:
        # New run: always add timestamp
        run_dirs = create_run_directories(runs_dir, args.run_name)
        actual_run_name = run_dirs['run_name']
    
    print(f"\nRun: {actual_run_name}")
    print(f"  Run directory: {run_dirs['run_dir']}")
    print(f"  Checkpoints: {run_dirs['checkpoints']}")
    print(f"  Results: {run_dirs['results']}")
    print(f"  Analytics: {run_dirs['analytics']}")
    print(f"  Wandb: {run_dirs['wandb']}")
    print(f"  Data: {run_dirs['data']}")
    
    # Copy data metadata (preprocessing metadata and split JSON)
    copy_data_metadata(
        Path(args.preprocessed_dir),
        Path(args.train_test_split_json),
        run_dirs['data']
    )
    
    # Copy config YAML to run's data directory
    copy_config(Path(initial_args.config), run_dirs['data'])
    
    # Setup device and print training info
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print_training_info(device, args.batch_size, args.epochs, args.lr)
    
    # Load data (two DataLoaders with early val prefetching)
    train_loader, val_loader, train_dataset, val_dataset = load_data(args)
    
    # Setup model and optimizer
    if resume_mode:
        # Resume: load everything from checkpoint (uses factories internally)
        model, criterion, optimizer, start_epoch, best_val_f1_macro, wandb_run_id = resume_from_checkpoint(
            Path(args.resume), device
        )
    else:
        # New training: create from config
        model, criterion, optimizer = setup_model_and_optimizer(args, device)
        start_epoch = 1
        best_val_f1_macro = 0.0
        wandb_run_id = None
    
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
        'run_name': actual_run_name,
    }
    
    # Initialize wandb (save to run_dirs['wandb'])
    # Use actual_run_name (with timestamp) for wandb run name
    args.device = device  # Add device to args for wandb
    config_path = Path(initial_args.config) if hasattr(initial_args, 'config') else None
    preprocessing_metadata_path = run_dirs['data'] / 'preprocessing_metadata.json'
    wandb_run_id, wandb_url = init_wandb_from_args(
        args, model, run_dirs['wandb'], 
        run_name=actual_run_name,
        group=getattr(args, 'group', None),
        resume_id=wandb_run_id,
        config_path=config_path,
        preprocessing_metadata_path=preprocessing_metadata_path
    )
    
    # Save wandb info (URL and run_id) to data directory
    import json
    wandb_info = {
        'run_id': wandb_run_id,
        'url': wandb_url,
        'run_name': actual_run_name,
        'group': getattr(args, 'group', None),
        'project': 'resnet3d-binary-classification'
    }
    wandb_info_path = run_dirs['data'] / 'wandb_info.json'
    with open(wandb_info_path, 'w', encoding='utf-8') as f:
        json.dump(wandb_info, f, indent=2)
    print(f"  Saved wandb info to: {wandb_info_path}")
    
    # ResultsTracker saves to run_dirs['results']
    results_tracker = ResultsTracker(results_config, run_dirs['results'], resume=resume_mode)
    
    # Trainer saves checkpoints to run_dirs['checkpoints']
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=run_dirs['checkpoints'],
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        results_tracker=results_tracker,
        best_val_f1_macro=best_val_f1_macro,
        wandb_run_id=wandb_run_id,
        in_channels=args.in_channels,
        num_classes=1
    )
    
    # Train
    summary = trainer.train(args.epochs, start_epoch=start_epoch)
    
    # Finalize results
    json_path = results_tracker.finalize(
        best_val_f1_macro=summary['best_val_f1_macro'],
        total_epochs=summary['total_epochs'],
        early_stopped=summary['early_stopped'],
        analytics_dir=run_dirs['analytics']
    )
    
    # Finalize wandb
    finalize_wandb(summary, has_validation=True)
    
    # Display final results
    print("\nTraining completed!")
    print_final_results(summary, run_dirs['results'])
    
    print(f"\nDetailed results saved to: {json_path}")
    print(f"\nRun directory: {run_dirs['run_dir']}")


if __name__ == "__main__":
    print("Starting training...")
    main()
