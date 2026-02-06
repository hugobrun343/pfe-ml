"""Checkpoint resuming utilities."""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))
sys.path.insert(0, str(scripts_dir / "helpers"))

from scripts.core.model import create_model
import torch.optim as optim


def resume_from_checkpoint(resume_path: Path, device: torch.device) -> Tuple[torch.nn.Module, nn.Module, optim.Optimizer, int, float, Optional[str]]:
    """
    Resume training from checkpoint using factories.
    Loads checkpoint once and creates model/optimizer automatically.
    
    Args:
        resume_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Tuple of (model, criterion, optimizer, start_epoch, best_val_f1_macro, wandb_run_id)
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If required hyperparameters are missing from checkpoint
    """
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    
    print(f"\nResuming from checkpoint: {resume_path}")
    
    # Load checkpoint once
    checkpoint = torch.load(str(resume_path), map_location=device)
    
    # Extract hyperparameters
    in_channels = checkpoint.get('in_channels')
    num_classes = checkpoint.get('num_classes')
    lr = checkpoint.get('learning_rate')
    weight_decay = checkpoint.get('weight_decay')
    
    if in_channels is None or num_classes is None:
        raise KeyError(
            f"Checkpoint missing required model hyperparameters. "
            f"Found: in_channels={in_channels}, num_classes={num_classes}"
        )
    
    if lr is None or weight_decay is None:
        raise KeyError(
            f"Checkpoint missing required optimizer hyperparameters. "
            f"Found: lr={lr}, weight_decay={weight_decay}"
        )
    
    # Create model
    model = create_model(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Loss function is always the same
    criterion = nn.BCEWithLogitsLoss()
    
    # Extract training state
    start_epoch = checkpoint['epoch'] + 1
    best_val_f1_macro = checkpoint.get('best_val_f1_macro', checkpoint.get('best_val_f1_class_1', 0.0))
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    print(f"  Resumed from epoch {checkpoint['epoch']}, continuing from epoch {start_epoch}")
    if wandb_run_id:
        print(f"  Wandb run ID: {wandb_run_id}")
    
    return model, criterion, optimizer, start_epoch, best_val_f1_macro, wandb_run_id
