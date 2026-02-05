"""
ResNet3D-50 for 3D Medical Image Classification

Architecture: ResNet-50 adapted for 3D inputs
- Input: (B, 3, 32, 256, 256) - 3-channel 3D patches
- Output: (B, num_classes) - classification logits

Layer configuration: [3, 4, 6, 3] bottleneck blocks
- Layer 1: 3 blocks, 256 channels
- Layer 2: 4 blocks, 512 channels  
- Layer 3: 6 blocks, 1024 channels
- Layer 4: 3 blocks, 2048 channels

Total parameters: ~46M (proper unique weights per block)

Dimension changes for input (B, 3, 32, 256, 256):
- Stem: (B, 3, 32, 256, 256) -> (B, 64, 8, 64, 64)
- Layer 1: (B, 64, 8, 64, 64) -> (B, 256, 8, 64, 64)
- Layer 2: (B, 256, 8, 64, 64) -> (B, 512, 4, 32, 32)
- Layer 3: (B, 512, 4, 32, 32) -> (B, 1024, 2, 16, 16)
- Layer 4: (B, 1024, 2, 16, 16) -> (B, 2048, 1, 8, 8)
- Head: (B, 2048, 1, 8, 8) -> (B, num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional


class ResNet3D50(nn.Module):
    """ResNet3D-50 for binary classification with unique weights per block"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1, initial_channels: int = 64):
        """
        Initialize ResNet3D-50
        
        Args:
            in_channels: Number of input channels (default: 3)
            num_classes: Number of output classes (default: 1 for binary)
            initial_channels: Number of channels after first conv layer (default: 64)
        """
        super(ResNet3D50, self).__init__()
        
        # ============================================================================
        # STEM: (B, 3, 32, 256, 256) -> (B, 64, 8, 64, 64)
        # ============================================================================
        
        # Initial convolution (7×7×7): 3 → 64, stride=2
        # (B, 3, 32, 256, 256) -> (B, 64, 16, 128, 128)
        self.conv1 = nn.Conv3d(in_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # MaxPooling (3×3×3): stride=2
        # (B, 64, 16, 128, 128) -> (B, 64, 8, 64, 64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ============================================================================
        # LAYER 1: (B, 64, 8, 64, 64) -> (B, 256, 8, 64, 64) - 3 blocks
        # ============================================================================
        
        # Block 1 (with projection): 64 -> 256 channels
        # conv1 (1×1×1): 64 → 64
        self.layer1_block1_conv1 = nn.Conv3d(64, 64, kernel_size=1, bias=False)
        self.layer1_block1_bn1 = nn.BatchNorm3d(64)
        # conv2 (3×3×3): 64 → 64
        self.layer1_block1_conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn2 = nn.BatchNorm3d(64)
        # conv3 (1×1×1): 64 → 256
        self.layer1_block1_conv3 = nn.Conv3d(64, 256, kernel_size=1, bias=False)
        self.layer1_block1_bn3 = nn.BatchNorm3d(256)
        # residual downsample (1×1×1): 64 → 256
        self.layer1_block1_downsample_conv = nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_block1_downsample_bn = nn.BatchNorm3d(256)

        # Blocks 2-3 (identity): 256 -> 256 channels - UNIQUE WEIGHTS per block
        # Each block: conv1 (1×1×1): 256 → 64, conv2 (3×3×3): 64 → 64, conv3 (1×1×1): 64 → 256
        self.layer1_blocks = nn.ModuleList()
        for i in range(2):  # 2 identity blocks
            self.layer1_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(256, 64, kernel_size=1, bias=False),      # 1×1×1: 256 → 64
                'bn1': nn.BatchNorm3d(64),
                'conv2': nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 64 → 64
                'bn2': nn.BatchNorm3d(64),
                'conv3': nn.Conv3d(64, 256, kernel_size=1, bias=False),      # 1×1×1: 64 → 256
                'bn3': nn.BatchNorm3d(256),
            }))
        
        # ============================================================================
        # LAYER 2: (B, 256, 8, 64, 64) -> (B, 512, 4, 32, 32) - 4 blocks
        # ============================================================================
        
        # Block 1 (with stride=2 downsample): 256 -> 512 channels
        # conv1 (1×1×1): 256 → 128
        self.layer2_block1_conv1 = nn.Conv3d(256, 128, kernel_size=1, bias=False)
        self.layer2_block1_bn1 = nn.BatchNorm3d(128)
        # conv2 (3×3×3): 128 → 128, stride=2
        self.layer2_block1_conv2 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_block1_bn2 = nn.BatchNorm3d(128)
        # conv3 (1×1×1): 128 → 512
        self.layer2_block1_conv3 = nn.Conv3d(128, 512, kernel_size=1, bias=False)
        self.layer2_block1_bn3 = nn.BatchNorm3d(512)
        # residual downsample (1×1×1): 256 → 512, stride=2
        self.layer2_block1_downsample_conv = nn.Conv3d(256, 512, kernel_size=1, stride=2, bias=False)
        self.layer2_block1_downsample_bn = nn.BatchNorm3d(512)

        # Blocks 2-4 (identity): 512 -> 512 channels - UNIQUE WEIGHTS per block
        # Each block: conv1 (1×1×1): 512 → 128, conv2 (3×3×3): 128 → 128, conv3 (1×1×1): 128 → 512
        self.layer2_blocks = nn.ModuleList()
        for i in range(3):  # 3 identity blocks
            self.layer2_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(512, 128, kernel_size=1, bias=False),     # 1×1×1: 512 → 128
                'bn1': nn.BatchNorm3d(128),
                'conv2': nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 128 → 128
                'bn2': nn.BatchNorm3d(128),
                'conv3': nn.Conv3d(128, 512, kernel_size=1, bias=False),     # 1×1×1: 128 → 512
                'bn3': nn.BatchNorm3d(512),
            }))

        # ============================================================================
        # LAYER 3: (B, 512, 4, 32, 32) -> (B, 1024, 2, 16, 16) - 6 blocks
        # ============================================================================
        
        # Block 1 (with stride=2 downsample): 512 -> 1024 channels
        # conv1 (1×1×1): 512 → 256
        self.layer3_block1_conv1 = nn.Conv3d(512, 256, kernel_size=1, bias=False)
        self.layer3_block1_bn1 = nn.BatchNorm3d(256)
        # conv2 (3×3×3): 256 → 256, stride=2
        self.layer3_block1_conv2 = nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_block1_bn2 = nn.BatchNorm3d(256)
        # conv3 (1×1×1): 256 → 1024
        self.layer3_block1_conv3 = nn.Conv3d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block1_bn3 = nn.BatchNorm3d(1024)
        # residual downsample (1×1×1): 512 → 1024, stride=2
        self.layer3_block1_downsample_conv = nn.Conv3d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.layer3_block1_downsample_bn = nn.BatchNorm3d(1024)

        # Blocks 2-6 (identity): 1024 -> 1024 channels - UNIQUE WEIGHTS per block
        # Each block: conv1 (1×1×1): 1024 → 256, conv2 (3×3×3): 256 → 256, conv3 (1×1×1): 256 → 1024
        self.layer3_blocks = nn.ModuleList()
        for i in range(5):  # 5 identity blocks
            self.layer3_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(1024, 256, kernel_size=1, bias=False),    # 1×1×1: 1024 → 256
                'bn1': nn.BatchNorm3d(256),
                'conv2': nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 256 → 256
                'bn2': nn.BatchNorm3d(256),
                'conv3': nn.Conv3d(256, 1024, kernel_size=1, bias=False),    # 1×1×1: 256 → 1024
                'bn3': nn.BatchNorm3d(1024),
            }))

        # ============================================================================
        # LAYER 4: (B, 1024, 2, 16, 16) -> (B, 2048, 1, 8, 8) - 3 blocks
        # ============================================================================
        
        # Block 1 (with stride=2 downsample): 1024 -> 2048 channels
        # conv1 (1×1×1): 1024 → 512
        self.layer4_block1_conv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)
        self.layer4_block1_bn1 = nn.BatchNorm3d(512)
        # conv2 (3×3×3): 512 → 512, stride=2
        self.layer4_block1_conv2 = nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_block1_bn2 = nn.BatchNorm3d(512)
        # conv3 (1×1×1): 512 → 2048
        self.layer4_block1_conv3 = nn.Conv3d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block1_bn3 = nn.BatchNorm3d(2048)
        # residual downsample (1×1×1): 1024 → 2048, stride=2
        self.layer4_block1_downsample_conv = nn.Conv3d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.layer4_block1_downsample_bn = nn.BatchNorm3d(2048)

        # Blocks 2-3 (identity): 2048 -> 2048 channels - UNIQUE WEIGHTS per block
        # Each block: conv1 (1×1×1): 2048 → 512, conv2 (3×3×3): 512 → 512, conv3 (1×1×1): 512 → 2048
        self.layer4_blocks = nn.ModuleList()
        for i in range(2):  # 2 identity blocks
            self.layer4_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(2048, 512, kernel_size=1, bias=False),    # 1×1×1: 2048 → 512
                'bn1': nn.BatchNorm3d(512),
                'conv2': nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 512 → 512
                'bn2': nn.BatchNorm3d(512),
                'conv3': nn.Conv3d(512, 2048, kernel_size=1, bias=False),    # 1×1×1: 512 → 2048
                'bn3': nn.BatchNorm3d(2048),
            }))

        # ============================================================================
        # HEAD: (B, 2048, 1, 8, 8) -> (B, num_classes)
        # ============================================================================
        
        # Global average pooling: (B, 2048, 1, 8, 8) -> (B, 2048, 1, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Flatten: (B, 2048, 1, 1, 1) -> (B, 2048)
        self.flatten = nn.Flatten()
        # Linear: (B, 2048) -> (B, num_classes)
        self.fc = nn.Linear(2048, num_classes)
    
    def _forward_block(self, x: torch.Tensor, block: nn.ModuleDict) -> torch.Tensor:
        """
        Forward pass through an identity block
        
        Args:
            x: Input tensor
            block: ModuleDict containing conv1, bn1, conv2, bn2, conv3, bn3
            
        Returns:
            Output tensor after residual addition
        """
        identity = x
        # conv1 (1×1×1) + BN + ReLU
        out = self.relu(block['bn1'](block['conv1'](x)))
        # conv2 (3×3×3) + BN + ReLU
        out = self.relu(block['bn2'](block['conv2'](out)))
        # conv3 (1×1×1) + BN
        out = block['bn3'](block['conv3'](out))
        # Residual connection + ReLU
        out = out + identity
        return self.relu(out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 3, depth, height, width)
               Expected: (B, 3, 32, 256, 256)
            
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # ============================== STEM ==============================
        # Initial convolution: (B, 3, 32, 256, 256) -> (B, 64, 16, 128, 128)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # MaxPooling: (B, 64, 16, 128, 128) -> (B, 64, 8, 64, 64)
        x = self.maxpool(x)

        # ============================ LAYER 1 =============================
        # (B, 64, 8, 64, 64) -> (B, 256, 8, 64, 64) - 3 blocks

        # Block 1 (projection shortcut)
        identity = self.layer1_block1_downsample_bn(self.layer1_block1_downsample_conv(x))
        out = self.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        out = self.relu(self.layer1_block1_bn2(self.layer1_block1_conv2(out)))
        out = self.layer1_block1_bn3(self.layer1_block1_conv3(out))
        x = self.relu(out + identity)
        
        # Blocks 2-3 (identity shortcut) - unique weights per block
        for block in self.layer1_blocks:
            x = self._forward_block(x, block)

        # ============================ LAYER 2 =============================
        # (B, 256, 8, 64, 64) -> (B, 512, 4, 32, 32) - 4 blocks

        # Block 1 (stride=2 downsample)
        identity = self.layer2_block1_downsample_bn(self.layer2_block1_downsample_conv(x))
        out = self.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        out = self.relu(self.layer2_block1_bn2(self.layer2_block1_conv2(out)))
        out = self.layer2_block1_bn3(self.layer2_block1_conv3(out))
        x = self.relu(out + identity)
        
        # Blocks 2-4 (identity shortcut) - unique weights per block
        for block in self.layer2_blocks:
            x = self._forward_block(x, block)

        # ============================ LAYER 3 =============================
        # (B, 512, 4, 32, 32) -> (B, 1024, 2, 16, 16) - 6 blocks

        # Block 1 (stride=2 downsample)
        identity = self.layer3_block1_downsample_bn(self.layer3_block1_downsample_conv(x))
        out = self.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        out = self.relu(self.layer3_block1_bn2(self.layer3_block1_conv2(out)))
        out = self.layer3_block1_bn3(self.layer3_block1_conv3(out))
        x = self.relu(out + identity)
        
        # Blocks 2-6 (identity shortcut) - unique weights per block
        for block in self.layer3_blocks:
            x = self._forward_block(x, block)

        # ============================ LAYER 4 =============================
        # (B, 1024, 2, 16, 16) -> (B, 2048, 1, 8, 8) - 3 blocks

        # Block 1 (stride=2 downsample)
        identity = self.layer4_block1_downsample_bn(self.layer4_block1_downsample_conv(x))
        out = self.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        out = self.relu(self.layer4_block1_bn2(self.layer4_block1_conv2(out)))
        out = self.layer4_block1_bn3(self.layer4_block1_conv3(out))
        x = self.relu(out + identity)
        
        # Blocks 2-3 (identity shortcut) - unique weights per block
        for block in self.layer4_blocks:
            x = self._forward_block(x, block)

        # ============================== HEAD ==============================
        # Global average pooling: (B, 2048, 1, 8, 8) -> (B, 2048, 1, 1, 1)
        x = self.avgpool(x)
        # Flatten: (B, 2048, 1, 1, 1) -> (B, 2048)
        x = self.flatten(x)
        # Linear: (B, 2048) -> (B, num_classes)
        x = self.fc(x)
        
        return x


def create_model(in_channels: int = 3, num_classes: int = 1) -> ResNet3D50:
    """
    Factory function to create ResNet3D-50 model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1)
        
    Returns:
        ResNet3D50 model instance
    """
    return ResNet3D50(in_channels=in_channels, num_classes=num_classes)
