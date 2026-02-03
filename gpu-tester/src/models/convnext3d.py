"""
ConvNeXt3D implementation for 3D medical image classification
Based on "A ConvNet for the 2020s" paper
Modern CNN architecture that rivals transformers
"""

import torch
import torch.nn as nn
from typing import List


class LayerNorm3D(nn.Module):
    """LayerNorm for 3D data (channels first)"""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super(LayerNorm3D, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt block for 3D"""
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6
    ):
        super(ConvNeXtBlock3D, self).__init__()
        
        # Depthwise convolution (7x7x7)
        self.dwconv = nn.Conv3d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )
        
        self.norm = LayerNorm3D(dim)
        
        # Pointwise/1x1x1 convolutions
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        input = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to (B, D, H, W, C) for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 4, 1)
        
        x = self.norm(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        
        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Residual connection
        x = input + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class ConvNeXt3D(nn.Module):
    """ConvNeXt3D architecture"""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6
    ):
        super(ConvNeXt3D, self).__init__()
        
        # Stem (patchify layer)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm3D(dims[0])
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm3D(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)
        
        # ConvNeXt blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock3D(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Classification head
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """Extract features"""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        
        # Global average pooling
        x = x.mean([-3, -2, -1])  # (B, C)
        
        # Classification head
        x = self.norm(x)
        x = self.head(x)
        
        return x


# ConvNeXt3D configurations
def get_convnext3d_tiny(in_channels: int = 1, num_classes: int = 2) -> ConvNeXt3D:
    """ConvNeXt3D-Tiny: ~28M params"""
    return ConvNeXt3D(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1
    )


def get_convnext3d_small(in_channels: int = 1, num_classes: int = 2) -> ConvNeXt3D:
    """ConvNeXt3D-Small: ~50M params"""
    return ConvNeXt3D(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2
    )


def get_convnext3d_base(in_channels: int = 1, num_classes: int = 2) -> ConvNeXt3D:
    """ConvNeXt3D-Base: ~89M params"""
    return ConvNeXt3D(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.3
    )


def get_convnext3d_large(in_channels: int = 1, num_classes: int = 2) -> ConvNeXt3D:
    """ConvNeXt3D-Large: ~197M params"""
    return ConvNeXt3D(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.4
    )


def get_convnext3d_xlarge(in_channels: int = 1, num_classes: int = 2) -> ConvNeXt3D:
    """ConvNeXt3D-XLarge: ~350M params"""
    return ConvNeXt3D(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        drop_path_rate=0.5
    )
