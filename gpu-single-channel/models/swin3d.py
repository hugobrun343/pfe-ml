"""
Swin Transformer 3D for medical image classification (via MONAI)

Input: (B, C, 32, 256, 256) - batch, channels, depth, height, width

Architecture: Swin Transformer adapted for 3D volumetric data (MONAI)
    - Patch Embedding: Conv3d patch_size=(2,2,2) stride
    - 4-stage hierarchical encoder with shifted-window attention
    - Patch Merging between stages (spatial halving, channel doubling)
    - Global Average Pooling on deepest feature map
    - Linear classification head

Available configs:
    Swin3D-Tiny:   embed_dim=48,  depths=[2,2,6,2], heads=[3,6,12,24]  ~10M params
    Swin3D-Small:  embed_dim=96,  depths=[2,2,6,2], heads=[3,6,12,24]  ~39M params

Feature map progression (Tiny, input 3x32x256x256):
    Stage 0: (B,  48, 16, 128, 128)
    Stage 1: (B,  96,  8,  64,  64)
    Stage 2: (B, 192,  4,  32,  32)
    Stage 3: (B, 384,  2,  16,  16)
    Stage 4: (B, 768,  1,   8,   8) -> GAP -> (B, 768) -> head -> (B, num_classes)
"""

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer


class Swin3DClassifier(nn.Module):
    """Swin Transformer 3D wrapper for classification.

    Uses MONAI's SwinTransformer backbone, takes the deepest feature map,
    applies global average pooling, and projects to num_classes logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 48,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: tuple = (4, 7, 7),
        patch_size: tuple = (2, 2, 2),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.backbone = SwinTransformer(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            spatial_dims=3,
        )

        # Deepest feature dim = embed_dim * 2^len(depths)
        deep_dim = embed_dim * (2 ** len(depths))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.norm = nn.LayerNorm(deep_dim)
        self.head = nn.Linear(deep_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)  e.g. (B, 3, 32, 256, 256)
        Returns:
            logits: (B, num_classes)
        """
        features = self.backbone(x)       # list of 5 feature maps
        x = features[-1]                  # deepest: (B, deep_dim, d, h, w)
        x = self.pool(x).flatten(1)       # (B, deep_dim)
        x = self.norm(x)                  # LayerNorm
        x = self.head(x)                  # (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_swin3d_tiny(in_channels: int = 3, num_classes: int = 1) -> Swin3DClassifier:
    """Swin3D-Tiny: embed_dim=48, ~10M params"""
    return Swin3DClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=48,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )


def create_swin3d_small(in_channels: int = 3, num_classes: int = 1) -> Swin3DClassifier:
    """Swin3D-Small: embed_dim=96, ~39M params"""
    return Swin3DClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )
