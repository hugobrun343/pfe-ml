"""
EfficientNet3D implementation for 3D medical image classification
Based on EfficientNet paper with compound scaling
"""

import torch
import torch.nn as nn
from typing import Optional, List
import math


class Swish(nn.Module):
    """Swish activation function (SiLU)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D"""
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SEBlock3D, self).__init__()
        reduced_channels = max(1, in_channels // reduction)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced_channels, kernel_size=1),
            Swish(),
            nn.Conv3d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConv3D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for 3D"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0
    ):
        super(MBConv3D, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        
        hidden_dim = in_channels * expand_ratio
        padding = kernel_size // 2
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                Swish()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm3d(hidden_dim),
            Swish()
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock3D(hidden_dim, reduction=int(1/se_ratio)))
        
        # Output projection
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        
        # Stochastic depth (drop connect)
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                keep_prob = 1 - self.drop_connect_rate
                random_tensor = keep_prob + torch.rand(
                    (x.shape[0], 1, 1, 1, 1),
                    dtype=x.dtype,
                    device=x.device
                )
                binary_tensor = torch.floor(random_tensor)
                out = out * binary_tensor / keep_prob
            out = out + x
        
        return out


class EfficientNet3D(nn.Module):
    """EfficientNet3D architecture"""
    
    def __init__(
        self,
        width_mult: float,
        depth_mult: float,
        in_channels: int = 1,
        num_classes: int = 2,
        drop_connect_rate: float = 0.2,
        dropout_rate: float = 0.2
    ):
        super(EfficientNet3D, self).__init__()
        
        # Base configuration [expand_ratio, channels, num_blocks, stride, kernel_size]
        base_config = [
            [1, 16, 1, 1, 3],   # Stage 1
            [6, 24, 2, 2, 3],   # Stage 2
            [6, 40, 2, 2, 5],   # Stage 3
            [6, 80, 3, 2, 3],   # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]
        
        # Initial stem
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            Swish()
        )
        
        # Build blocks
        blocks = []
        total_blocks = sum([self._round_repeats(config[2], depth_mult) for config in base_config])
        block_idx = 0
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in base_config:
            out_channels = self._round_filters(channels, width_mult)
            num_blocks = self._round_repeats(num_blocks, depth_mult)
            
            for i in range(num_blocks):
                # Calculate drop connect rate
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                blocks.append(MBConv3D(
                    in_channels=out_channels if i > 0 else self._get_prev_channels(blocks),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    drop_connect_rate=drop_rate
                ))
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv3d(self._round_filters(320, width_mult), final_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(final_channels),
            Swish(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_channels, num_classes)
        )
        
        self._initialize_weights()
    
    def _get_prev_channels(self, blocks: List[nn.Module]) -> int:
        """Get output channels from previous block"""
        if len(blocks) == 0:
            return self.stem[0].out_channels
        for layer in reversed(blocks[-1].conv):
            if isinstance(layer, nn.Conv3d):
                return layer.out_channels
        return self.stem[0].out_channels
    
    @staticmethod
    def _round_filters(filters: int, width_mult: float, divisor: int = 8) -> int:
        """Round number of filters based on width multiplier"""
        filters = int(filters * width_mult)
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return new_filters
    
    @staticmethod
    def _round_repeats(repeats: int, depth_mult: float) -> int:
        """Round number of repeats based on depth multiplier"""
        return int(math.ceil(repeats * depth_mult))
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# EfficientNet configurations
def get_efficientnet3d_b0(in_channels: int = 1, num_classes: int = 2) -> EfficientNet3D:
    """EfficientNet3D-B0: baseline"""
    return EfficientNet3D(
        width_mult=1.0,
        depth_mult=1.0,
        in_channels=in_channels,
        num_classes=num_classes,
        drop_connect_rate=0.2
    )


def get_efficientnet3d_b1(in_channels: int = 1, num_classes: int = 2) -> EfficientNet3D:
    """EfficientNet3D-B1: slightly larger"""
    return EfficientNet3D(
        width_mult=1.0,
        depth_mult=1.1,
        in_channels=in_channels,
        num_classes=num_classes,
        drop_connect_rate=0.2
    )


def get_efficientnet3d_b2(in_channels: int = 1, num_classes: int = 2) -> EfficientNet3D:
    """EfficientNet3D-B2"""
    return EfficientNet3D(
        width_mult=1.1,
        depth_mult=1.2,
        in_channels=in_channels,
        num_classes=num_classes,
        drop_connect_rate=0.3
    )


def get_efficientnet3d_b3(in_channels: int = 1, num_classes: int = 2) -> EfficientNet3D:
    """EfficientNet3D-B3"""
    return EfficientNet3D(
        width_mult=1.2,
        depth_mult=1.4,
        in_channels=in_channels,
        num_classes=num_classes,
        drop_connect_rate=0.3
    )


def get_efficientnet3d_b4(in_channels: int = 1, num_classes: int = 2) -> EfficientNet3D:
    """EfficientNet3D-B4"""
    return EfficientNet3D(
        width_mult=1.4,
        depth_mult=1.8,
        in_channels=in_channels,
        num_classes=num_classes,
        drop_connect_rate=0.4
    )
