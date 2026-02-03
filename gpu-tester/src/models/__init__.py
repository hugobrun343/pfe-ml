"""
3D Deep Learning Models for Medical Image Classification
=========================================================

This package provides multiple state-of-the-art architectures for 3D medical image classification:

Architectures:
- ResNet3D: Standard ResNet adapted for 3D (variants: 10, 14, 18, 26, 34, 50, 101, 152)
- EfficientNet3D: Efficient architecture with compound scaling (variants: B0-B4)
- ViT3D: Vision Transformer for 3D volumes (variants: Tiny, Small, Base, Large)
- ConvNeXt3D: Modern CNN architecture (variants: Tiny, Small, Base, Large, XLarge)
- SE-ResNet3D: ResNet with Squeeze-and-Excitation attention (variants: 18, 34, 50, 101, 152)

Features:
- Mixed precision training (FP16/BF16) for ~50% VRAM reduction
- Gradient checkpointing for 30-50% VRAM savings
- Comprehensive model utilities (parameter counting, VRAM estimation, etc.)

Usage:
    from models import get_model_by_name, get_available_models
    
    # Get a model
    model = get_model_by_name("ResNet3D-34", in_channels=1, num_classes=2)
    
    # List all available models
    all_models = get_available_models()
    
    # Enable gradient checkpointing for VRAM savings
    model = get_model_by_name("ResNet3D-50", use_gradient_checkpointing=True)
    
    # Use mixed precision
    from models import MixedPrecisionWrapper
    mp_model = MixedPrecisionWrapper(model, dtype="float16")
"""

import warnings
from typing import List, Optional
import torch.nn as nn

# Import ResNet3D
from .resnet3d import (
    ResNet3D,
    BasicBlock3D,
    Bottleneck3D,
    get_resnet3d_10,
    get_resnet3d_18,
    get_resnet3d_34,
    get_resnet3d_50,
    get_resnet3d_101,
)

# Import EfficientNet3D
try:
    from .efficientnet3d import (
        EfficientNet3D,
        get_efficientnet3d_b0,
        get_efficientnet3d_b1,
        get_efficientnet3d_b2,
        get_efficientnet3d_b3,
        get_efficientnet3d_b4,
    )
    EFFICIENTNET_AVAILABLE = True
except ImportError as e:
    EFFICIENTNET_AVAILABLE = False
    warnings.warn(f"EfficientNet3D not available: {e}")

# Import ViT3D
try:
    from .vit3d import (
        VisionTransformer3D,
        get_vit3d_tiny,
        get_vit3d_small,
        get_vit3d_base,
        get_vit3d_large,
    )
    VIT_AVAILABLE = True
except ImportError as e:
    VIT_AVAILABLE = False
    warnings.warn(f"ViT3D not available: {e}")

# Import ConvNeXt3D
try:
    from .convnext3d import (
        ConvNeXt3D,
        get_convnext3d_tiny,
        get_convnext3d_small,
        get_convnext3d_base,
        get_convnext3d_large,
        get_convnext3d_xlarge,
    )
    CONVNEXT_AVAILABLE = True
except ImportError as e:
    CONVNEXT_AVAILABLE = False
    warnings.warn(f"ConvNeXt3D not available: {e}")

# Import SE-ResNet3D
try:
    from .seresnet3d import (
        SEResNet3D,
        SEBasicBlock3D,
        SEBottleneck3D,
        get_seresnet3d_18,
        get_seresnet3d_34,
        get_seresnet3d_50,
        get_seresnet3d_101,
        get_seresnet3d_152,
    )
    SERESNET_AVAILABLE = True
except ImportError as e:
    SERESNET_AVAILABLE = False
    warnings.warn(f"SE-ResNet3D not available: {e}")

# Import utilities
from .utils import (
    count_parameters,
    get_model_info,
    apply_gradient_checkpointing,
    MixedPrecisionWrapper,
    format_bytes,
    estimate_vram,
)


def get_model_by_name(
    model_name: str,
    in_channels: int = 1,
    num_classes: int = 2,
    use_gradient_checkpointing: bool = False,
    **kwargs
) -> nn.Module:
    """
    Get model by name with optional optimizations
    
    Args:
        model_name: Model name (e.g., "ResNet3D-34", "EfficientNet3D-B0", "ViT3D-Base")
        in_channels: Number of input channels (default: 1 for grayscale medical images)
        num_classes: Number of output classes (default: 2 for binary classification)
        use_gradient_checkpointing: Enable gradient checkpointing to save VRAM (default: False)
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        PyTorch model instance
        
    Available Models:
        ResNet3D: ResNet3D-10, ResNet3D-18, ResNet3D-34, ResNet3D-50, ResNet3D-101
        EfficientNet3D: EfficientNet3D-B0, EfficientNet3D-B1, EfficientNet3D-B2,
                        EfficientNet3D-B3, EfficientNet3D-B4
        ViT3D: ViT3D-Tiny, ViT3D-Small, ViT3D-Base, ViT3D-Large
        ConvNeXt3D: ConvNeXt3D-Tiny, ConvNeXt3D-Small, ConvNeXt3D-Base,
                    ConvNeXt3D-Large, ConvNeXt3D-XLarge
        SE-ResNet3D: SE-ResNet3D-18, SE-ResNet3D-34, SE-ResNet3D-50,
                     SE-ResNet3D-101, SE-ResNet3D-152
    
    Example:
        >>> model = get_model_by_name("ResNet3D-34", in_channels=1, num_classes=2)
        >>> model = get_model_by_name("ViT3D-Base", use_gradient_checkpointing=True)
    """
    model_factory = {
        # ResNet3D variants (5)
        "ResNet3D-10": get_resnet3d_10,
        "ResNet3D-18": get_resnet3d_18,
        "ResNet3D-34": get_resnet3d_34,
        "ResNet3D-50": get_resnet3d_50,
        "ResNet3D-101": get_resnet3d_101,
    }
    
    # Add EfficientNet3D variants
    if EFFICIENTNET_AVAILABLE:
        model_factory.update({
            "EfficientNet3D-B0": get_efficientnet3d_b0,
            "EfficientNet3D-B1": get_efficientnet3d_b1,
            "EfficientNet3D-B2": get_efficientnet3d_b2,
            "EfficientNet3D-B3": get_efficientnet3d_b3,
            "EfficientNet3D-B4": get_efficientnet3d_b4,
        })
    
    # Add ViT3D variants
    if VIT_AVAILABLE:
        model_factory.update({
            "ViT3D-Tiny": get_vit3d_tiny,
            "ViT3D-Small": get_vit3d_small,
            "ViT3D-Base": get_vit3d_base,
            "ViT3D-Large": get_vit3d_large,
        })
    
    # Add ConvNeXt3D variants
    if CONVNEXT_AVAILABLE:
        model_factory.update({
            "ConvNeXt3D-Tiny": get_convnext3d_tiny,
            "ConvNeXt3D-Small": get_convnext3d_small,
            "ConvNeXt3D-Base": get_convnext3d_base,
            "ConvNeXt3D-Large": get_convnext3d_large,
            "ConvNeXt3D-XLarge": get_convnext3d_xlarge,
        })
    
    # Add SE-ResNet3D variants
    if SERESNET_AVAILABLE:
        model_factory.update({
            "SE-ResNet3D-18": get_seresnet3d_18,
            "SE-ResNet3D-34": get_seresnet3d_34,
            "SE-ResNet3D-50": get_seresnet3d_50,
            "SE-ResNet3D-101": get_seresnet3d_101,
            "SE-ResNet3D-152": get_seresnet3d_152,
        })
    
    if model_name not in model_factory:
        available = get_available_models()
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models ({len(available)}): {', '.join(available)}"
        )
    
    # Create model
    model = model_factory[model_name](in_channels, num_classes, **kwargs)
    
    # Apply gradient checkpointing if requested
    if use_gradient_checkpointing:
        model = apply_gradient_checkpointing(model)
    
    return model


def get_available_models() -> List[str]:
    """
    Get list of all available model names
    
    Returns:
        List of model names sorted by family
        
    Example:
        >>> models = get_available_models()
        >>> print(f"Total models: {len(models)}")
        >>> for model in models:
        ...     print(model)
    """
    models = [
        "ResNet3D-10", "ResNet3D-18", "ResNet3D-34", "ResNet3D-50", "ResNet3D-101"
    ]
    
    if EFFICIENTNET_AVAILABLE:
        models.extend([
            "EfficientNet3D-B0", "EfficientNet3D-B1", "EfficientNet3D-B2",
            "EfficientNet3D-B3", "EfficientNet3D-B4"
        ])
    
    if VIT_AVAILABLE:
        models.extend([
            "ViT3D-Tiny", "ViT3D-Small", "ViT3D-Base", "ViT3D-Large"
        ])
    
    if CONVNEXT_AVAILABLE:
        models.extend([
            "ConvNeXt3D-Tiny", "ConvNeXt3D-Small", "ConvNeXt3D-Base",
            "ConvNeXt3D-Large", "ConvNeXt3D-XLarge"
        ])
    
    if SERESNET_AVAILABLE:
        models.extend([
            "SE-ResNet3D-18", "SE-ResNet3D-34", "SE-ResNet3D-50",
            "SE-ResNet3D-101", "SE-ResNet3D-152"
        ])
    
    return models


def get_model_families() -> dict:
    """
    Get models organized by family
    
    Returns:
        Dictionary mapping family names to lists of model names
        
    Example:
        >>> families = get_model_families()
        >>> for family, models in families.items():
        ...     print(f"{family}: {len(models)} models")
    """
    families = {
        "ResNet3D": [
            "ResNet3D-10", "ResNet3D-18", "ResNet3D-34", "ResNet3D-50", "ResNet3D-101"
        ]
    }
    
    if EFFICIENTNET_AVAILABLE:
        families["EfficientNet3D"] = [
            "EfficientNet3D-B0", "EfficientNet3D-B1", "EfficientNet3D-B2",
            "EfficientNet3D-B3", "EfficientNet3D-B4"
        ]
    
    if VIT_AVAILABLE:
        families["ViT3D"] = [
            "ViT3D-Tiny", "ViT3D-Small", "ViT3D-Base", "ViT3D-Large"
        ]
    
    if CONVNEXT_AVAILABLE:
        families["ConvNeXt3D"] = [
            "ConvNeXt3D-Tiny", "ConvNeXt3D-Small", "ConvNeXt3D-Base",
            "ConvNeXt3D-Large", "ConvNeXt3D-XLarge"
        ]
    
    if SERESNET_AVAILABLE:
        families["SE-ResNet3D"] = [
            "SE-ResNet3D-18", "SE-ResNet3D-34", "SE-ResNet3D-50",
            "SE-ResNet3D-101", "SE-ResNet3D-152"
        ]
    
    return families


# Export all public APIs
__all__ = [
    # Main functions
    'get_model_by_name',
    'get_available_models',
    'get_model_families',
    
    # Utility functions
    'count_parameters',
    'get_model_info',
    'apply_gradient_checkpointing',
    'MixedPrecisionWrapper',
    'format_bytes',
    'estimate_vram',
    
    # ResNet3D
    'ResNet3D',
    'BasicBlock3D',
    'Bottleneck3D',
    'get_resnet3d_10',
    'get_resnet3d_18',
    'get_resnet3d_34',
    'get_resnet3d_50',
    'get_resnet3d_101',
]

# Conditionally add other architectures
if EFFICIENTNET_AVAILABLE:
    __all__.extend([
        'EfficientNet3D',
        'get_efficientnet3d_b0',
        'get_efficientnet3d_b1',
        'get_efficientnet3d_b2',
        'get_efficientnet3d_b3',
        'get_efficientnet3d_b4',
    ])

if VIT_AVAILABLE:
    __all__.extend([
        'VisionTransformer3D',
        'get_vit3d_tiny',
        'get_vit3d_small',
        'get_vit3d_base',
        'get_vit3d_large',
    ])

if CONVNEXT_AVAILABLE:
    __all__.extend([
        'ConvNeXt3D',
        'get_convnext3d_tiny',
        'get_convnext3d_small',
        'get_convnext3d_base',
        'get_convnext3d_large',
        'get_convnext3d_xlarge',
    ])

if SERESNET_AVAILABLE:
    __all__.extend([
        'SEResNet3D',
        'SEBasicBlock3D',
        'SEBottleneck3D',
        'get_seresnet3d_18',
        'get_seresnet3d_34',
        'get_seresnet3d_50',
        'get_seresnet3d_101',
        'get_seresnet3d_152',
    ])


# Version info
__version__ = "1.0.0"
__author__ = "Your Team"
__description__ = "3D Deep Learning Models for Medical Image Classification"
