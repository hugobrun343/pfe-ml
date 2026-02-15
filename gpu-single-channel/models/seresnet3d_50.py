"""
SE-ResNet3D-50 for 3D Medical Image Classification

Architecture: ResNet-50 with Squeeze-and-Excitation blocks adapted for 3D inputs
- Input: (B, 3, 32, 256, 256) - 3-channel 3D patches
- Output: (B, num_classes) - classification logits

SE (Squeeze-and-Excitation) mechanism:
- Squeeze: Global average pooling to get channel descriptor
- Excitation: Two FC layers to learn channel attention weights
- Scale: Multiply features by learned attention weights

Layer configuration: [3, 4, 6, 3] bottleneck blocks with SE attention
- Layer 1: 3 SE-blocks, 256 channels
- Layer 2: 4 SE-blocks, 512 channels  
- Layer 3: 6 SE-blocks, 1024 channels
- Layer 4: 3 SE-blocks, 2048 channels

Total parameters: ~49M (ResNet-50 + SE overhead)

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


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps
    
    Operations:
    1. Squeeze: Global average pooling (B, C, D, H, W) -> (B, C)
    2. Excitation: FC -> ReLU -> FC -> Sigmoid (B, C) -> (B, C)
    3. Scale: Element-wise multiply (B, C, D, H, W) * (B, C, 1, 1, 1)
    
    Args:
        channels: Number of input/output channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)  # Minimum 8 channels
        
        # Squeeze: global average pooling
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        # FC1: channels → reduced
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # FC2: reduced → channels
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size()[:2]
        # Squeeze: (B, C, D, H, W) -> (B, C)
        y = self.squeeze(x).view(b, c)
        # Excitation: (B, C) -> (B, C)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).view(b, c, 1, 1, 1)
        # Scale: (B, C, D, H, W) * (B, C, 1, 1, 1) -> (B, C, D, H, W)
        return x * y


class SEResNet3D50(nn.Module):
    """SE-ResNet3D-50 for binary classification with unique weights per block"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        """
        Initialize SE-ResNet3D-50
        
        Args:
            in_channels: Number of input channels (default: 3)
            num_classes: Number of output classes (default: 1 for binary)
        """
        super().__init__()
        
        # ============================================================================
        # STEM: (B, 3, 32, 256, 256) -> (B, 64, 8, 64, 64)
        # ============================================================================
        
        # Initial convolution (7×7×7): 3 → 64, stride=2
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPooling (3×3×3): stride=2
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ============================================================================
        # LAYER 1: (B, 64, 8, 64, 64) -> (B, 256, 8, 64, 64) - 3 SE-blocks
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
        # SE block: channel attention on 256 channels
        self.layer1_block1_se = SEBlock3D(256)
        # residual downsample (1×1×1): 64 → 256
        self.layer1_block1_downsample_conv = nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_block1_downsample_bn = nn.BatchNorm3d(256)

        # Blocks 2-3 (identity): 256 -> 256 channels - UNIQUE WEIGHTS per block
        # Each block: conv1 (1×1×1): 256 → 64, conv2 (3×3×3): 64 → 64, conv3 (1×1×1): 64 → 256, SE
        self.layer1_blocks = nn.ModuleList()
        for i in range(2):
            self.layer1_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(256, 64, kernel_size=1, bias=False),      # 1×1×1: 256 → 64
                'bn1': nn.BatchNorm3d(64),
                'conv2': nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 64 → 64
                'bn2': nn.BatchNorm3d(64),
                'conv3': nn.Conv3d(64, 256, kernel_size=1, bias=False),      # 1×1×1: 64 → 256
                'bn3': nn.BatchNorm3d(256),
                'se': SEBlock3D(256),                                         # SE attention
            }))
        
        # ============================================================================
        # LAYER 2: (B, 256, 8, 64, 64) -> (B, 512, 4, 32, 32) - 4 SE-blocks
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
        # SE block: channel attention on 512 channels
        self.layer2_block1_se = SEBlock3D(512)
        # residual downsample (1×1×1): 256 → 512, stride=2
        self.layer2_block1_downsample_conv = nn.Conv3d(256, 512, kernel_size=1, stride=2, bias=False)
        self.layer2_block1_downsample_bn = nn.BatchNorm3d(512)

        # Blocks 2-4 (identity): 512 -> 512 channels - UNIQUE WEIGHTS per block
        self.layer2_blocks = nn.ModuleList()
        for i in range(3):
            self.layer2_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(512, 128, kernel_size=1, bias=False),     # 1×1×1: 512 → 128
                'bn1': nn.BatchNorm3d(128),
                'conv2': nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 128 → 128
                'bn2': nn.BatchNorm3d(128),
                'conv3': nn.Conv3d(128, 512, kernel_size=1, bias=False),     # 1×1×1: 128 → 512
                'bn3': nn.BatchNorm3d(512),
                'se': SEBlock3D(512),                                         # SE attention
            }))

        # ============================================================================
        # LAYER 3: (B, 512, 4, 32, 32) -> (B, 1024, 2, 16, 16) - 6 SE-blocks
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
        # SE block: channel attention on 1024 channels
        self.layer3_block1_se = SEBlock3D(1024)
        # residual downsample (1×1×1): 512 → 1024, stride=2
        self.layer3_block1_downsample_conv = nn.Conv3d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.layer3_block1_downsample_bn = nn.BatchNorm3d(1024)

        # Blocks 2-6 (identity): 1024 -> 1024 channels - UNIQUE WEIGHTS per block
        self.layer3_blocks = nn.ModuleList()
        for i in range(5):
            self.layer3_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(1024, 256, kernel_size=1, bias=False),    # 1×1×1: 1024 → 256
                'bn1': nn.BatchNorm3d(256),
                'conv2': nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 256 → 256
                'bn2': nn.BatchNorm3d(256),
                'conv3': nn.Conv3d(256, 1024, kernel_size=1, bias=False),    # 1×1×1: 256 → 1024
                'bn3': nn.BatchNorm3d(1024),
                'se': SEBlock3D(1024),                                        # SE attention
            }))

        # ============================================================================
        # LAYER 4: (B, 1024, 2, 16, 16) -> (B, 2048, 1, 8, 8) - 3 SE-blocks
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
        # SE block: channel attention on 2048 channels
        self.layer4_block1_se = SEBlock3D(2048)
        # residual downsample (1×1×1): 1024 → 2048, stride=2
        self.layer4_block1_downsample_conv = nn.Conv3d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.layer4_block1_downsample_bn = nn.BatchNorm3d(2048)

        # Blocks 2-3 (identity): 2048 -> 2048 channels - UNIQUE WEIGHTS per block
        self.layer4_blocks = nn.ModuleList()
        for i in range(2):
            self.layer4_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv3d(2048, 512, kernel_size=1, bias=False),    # 1×1×1: 2048 → 512
                'bn1': nn.BatchNorm3d(512),
                'conv2': nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3×3: 512 → 512
                'bn2': nn.BatchNorm3d(512),
                'conv3': nn.Conv3d(512, 2048, kernel_size=1, bias=False),    # 1×1×1: 512 → 2048
                'bn3': nn.BatchNorm3d(2048),
                'se': SEBlock3D(2048),                                        # SE attention
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
        Forward pass through an SE-identity block
        
        Structure: conv1 -> BN -> ReLU -> conv2 -> BN -> ReLU -> conv3 -> BN -> SE -> add -> ReLU
        """
        identity = x
        # conv1 (1×1×1) + BN + ReLU
        out = self.relu(block['bn1'](block['conv1'](x)))
        # conv2 (3×3×3) + BN + ReLU
        out = self.relu(block['bn2'](block['conv2'](out)))
        # conv3 (1×1×1) + BN
        out = block['bn3'](block['conv3'](out))
        # SE attention
        out = block['se'](out)
        # Residual connection + ReLU
        return self.relu(out + identity)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 3, 32, 256, 256)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # ============================== STEM ==============================
        # (B, 3, 32, 256, 256) -> (B, 64, 8, 64, 64)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # ============================ LAYER 1 =============================
        # (B, 64, 8, 64, 64) -> (B, 256, 8, 64, 64) - 3 SE-blocks
        
        # Block 1 (projection shortcut)
        identity = self.layer1_block1_downsample_bn(self.layer1_block1_downsample_conv(x))
        out = self.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        out = self.relu(self.layer1_block1_bn2(self.layer1_block1_conv2(out)))
        out = self.layer1_block1_se(self.layer1_block1_bn3(self.layer1_block1_conv3(out)))
        x = self.relu(out + identity)
        
        # Blocks 2-3 (identity) - unique weights
        for block in self.layer1_blocks:
            x = self._forward_block(x, block)

        # ============================ LAYER 2 =============================
        # (B, 256, 8, 64, 64) -> (B, 512, 4, 32, 32) - 4 SE-blocks
        
        # Block 1 (stride=2 downsample)
        identity = self.layer2_block1_downsample_bn(self.layer2_block1_downsample_conv(x))
        out = self.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        out = self.relu(self.layer2_block1_bn2(self.layer2_block1_conv2(out)))
        out = self.layer2_block1_se(self.layer2_block1_bn3(self.layer2_block1_conv3(out)))
        x = self.relu(out + identity)
        
        # Blocks 2-4 (identity) - unique weights
        for block in self.layer2_blocks:
            x = self._forward_block(x, block)

        # ============================ LAYER 3 =============================
        # (B, 512, 4, 32, 32) -> (B, 1024, 2, 16, 16) - 6 SE-blocks
        
        # Block 1 (stride=2 downsample)
        identity = self.layer3_block1_downsample_bn(self.layer3_block1_downsample_conv(x))
        out = self.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        out = self.relu(self.layer3_block1_bn2(self.layer3_block1_conv2(out)))
        out = self.layer3_block1_se(self.layer3_block1_bn3(self.layer3_block1_conv3(out)))
        x = self.relu(out + identity)
        
        # Blocks 2-6 (identity) - unique weights
        for block in self.layer3_blocks:
            x = self._forward_block(x, block)

        # ============================ LAYER 4 =============================
        # (B, 1024, 2, 16, 16) -> (B, 2048, 1, 8, 8) - 3 SE-blocks
        
        # Block 1 (stride=2 downsample)
        identity = self.layer4_block1_downsample_bn(self.layer4_block1_downsample_conv(x))
        out = self.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        out = self.relu(self.layer4_block1_bn2(self.layer4_block1_conv2(out)))
        out = self.layer4_block1_se(self.layer4_block1_bn3(self.layer4_block1_conv3(out)))
        x = self.relu(out + identity)
        
        # Blocks 2-3 (identity) - unique weights
        for block in self.layer4_blocks:
            x = self._forward_block(x, block)

        # ============================== HEAD ==============================
        # (B, 2048, 1, 8, 8) -> (B, num_classes)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def create_model(in_channels: int = 3, num_classes: int = 1) -> SEResNet3D50:
    """Factory function to create SE-ResNet3D-50 model"""
    return SEResNet3D50(in_channels=in_channels, num_classes=num_classes)
