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

# Import model registry from new models module
from scripts.models import get_model, MODEL_REGISTRY


def setup_model_and_optimizer(args: Namespace, device: torch.device):
    """
    Create model, loss function and optimizer.
    
    Args:
        args: Parsed command line arguments containing:
            - model_name: Name of model to use (e.g., 'resnet3d_50', 'vit3d_base')
            - in_channels: Number of input channels
            - lr: Learning rate
            - weight_decay: Weight decay for optimizer
        device: Device to use (cuda/cpu)
        
    Returns:
        Tuple of (model, criterion, optimizer)
        
    Available models:
        - resnet3d_50: ResNet3D-50 (~25M params)
        - resnet3d_101: ResNet3D-101 (~44M params)
        - seresnet3d_50: SE-ResNet3D-50 (~28M params)
        - seresnet3d_101: SE-ResNet3D-101 (~49M params)
        - vit3d_base: ViT3D-Base (~86M params)
        - vit3d_large: ViT3D-Large (~304M params)
        - convnext3d_small: ConvNeXt3D-Small (~50M params)
        - convnext3d_large: ConvNeXt3D-Large (~198M params)
    """
    # Get model name from args (default: resnet3d_50 for backward compatibility)
    model_name = getattr(args, 'model_name', 'resnet3d_50')
    in_channels = getattr(args, 'in_channels', 3)
    
    print(f"  Model: {model_name}")
    print(f"  Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Create model using registry
    model = get_model(model_name, in_channels=in_channels, num_classes=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function (Binary Cross-Entropy with Logits)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer (Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    return model, criterion, optimizer


def list_available_models():
    """Print list of available models."""
    print("Available models:")
    for name in MODEL_REGISTRY.keys():
        print(f"  - {name}")
