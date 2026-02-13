"""
3D Models for medical image classification

Available models:
- ResNet3D-50: resnet3d_50.py
- ResNet3D-101: resnet3d_101.py
- SE-ResNet3D-50: seresnet3d_50.py
- SE-ResNet3D-101: seresnet3d_101.py
- ViT3D-Base: vit3d_base.py
- ViT3D-Large: vit3d_large.py
- ConvNeXt3D-Small: convnext3d_small.py
- ConvNeXt3D-Large: convnext3d_large.py
- Swin3D-Tiny: swin3d.py (MONAI, ~10M params)
- Swin3D-Small: swin3d.py (MONAI, ~39M params)
- DenseNet121-3D: densenet3d.py (MONAI, ~11M params)
"""

from .resnet3d_50 import ResNet3D50, create_model as create_resnet3d_50
from .resnet3d_101 import ResNet3D101, create_model as create_resnet3d_101
from .seresnet3d_50 import SEResNet3D50, create_model as create_seresnet3d_50
from .seresnet3d_101 import SEResNet3D101, create_model as create_seresnet3d_101
from .vit3d_base import ViT3DBase, create_model as create_vit3d_base
from .vit3d_large import ViT3DLarge, create_model as create_vit3d_large
from .convnext3d_small import ConvNeXt3DSmall, create_model as create_convnext3d_small
from .convnext3d_large import ConvNeXt3DLarge, create_model as create_convnext3d_large
from .swin3d import Swin3DClassifier, create_swin3d_tiny, create_swin3d_small
from .densenet3d import DenseNet3D121, create_model as create_densenet3d_121


MODEL_REGISTRY = {
    'resnet3d_50': create_resnet3d_50,
    'resnet3d_101': create_resnet3d_101,
    'seresnet3d_50': create_seresnet3d_50,
    'seresnet3d_101': create_seresnet3d_101,
    'vit3d_base': create_vit3d_base,
    'vit3d_large': create_vit3d_large,
    'convnext3d_small': create_convnext3d_small,
    'convnext3d_large': create_convnext3d_large,
    'swin3d_tiny': create_swin3d_tiny,
    'swin3d_small': create_swin3d_small,
    'densenet3d_121': create_densenet3d_121,
}


def get_model(model_name: str, in_channels: int = 3, num_classes: int = 1):
    """Factory function to get model by name
    
    Args:
        model_name: One of the registered model names
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1 for binary)
    
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](in_channels=in_channels, num_classes=num_classes)


__all__ = [
    'ResNet3D50', 'ResNet3D101',
    'SEResNet3D50', 'SEResNet3D101',
    'ViT3DBase', 'ViT3DLarge',
    'ConvNeXt3DSmall', 'ConvNeXt3DLarge',
    'Swin3DClassifier',
    'DenseNet3D121',
    'get_model', 'MODEL_REGISTRY',
]
