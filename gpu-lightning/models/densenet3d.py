"""
DenseNet121-3D for medical image classification (via MONAI)

Input: (B, C, 32, 256, 256) - batch, channels, depth, height, width

Architecture: DenseNet with dense connections between all layers
    - Each layer receives features from ALL previous layers (concatenation)
    - 4 Dense Blocks: 6, 12, 24, 16 layers  (= 121 total)
    - Transition layers between blocks (1x1 conv + pooling)
    - Global Average Pooling + Linear classification head

Key advantage: feature reuse + strong gradient flow → good on small datasets

Total parameters: ~11M
"""

import torch.nn as nn
from monai.networks.nets import DenseNet121


class DenseNet3D121(nn.Module):
    """DenseNet121-3D wrapper for classification via MONAI."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
        )

    def forward(self, x):
        return self.model(x)


def create_model(in_channels: int = 3, num_classes: int = 1) -> DenseNet3D121:
    """Factory function to create DenseNet121-3D model."""
    return DenseNet3D121(in_channels=in_channels, num_classes=num_classes)
