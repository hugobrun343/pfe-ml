"""Model and optimizer setup utilities."""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Namespace
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))

from scripts.core.model import create_model


def setup_model_and_optimizer(args: Namespace, device: torch.device):
    """
    Create model, loss function and optimizer.
    
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
