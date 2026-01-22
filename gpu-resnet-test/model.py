import torch
import torch.nn as nn
from typing import Optional


class ResNet3D50(nn.Module):
    """ResNet3D-50 for binary classification"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1, initial_channels: int = 64):
        """
        Initialize ResNet3D-50
        
        Args:
            in_channels: Number of input channels (default: 3)
            num_classes: Number of output classes (default: 1 for binary)
            initial_channels: Number of channels after first conv layer (default: 64)
        """
        super(ResNet3D50, self).__init__()
        
        # Initial convolution: (B, 3, D, H, W) -> (B, 64, D/2, H/2, W/2)
        self.conv1 = nn.Conv3d(in_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # MaxPooling: (B, 64, D/2, H/2, W/2) -> (B, 64, D/4, H/4, W/4)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        ## Layer 1: (B, 64, D/4, H/4, W/4) -> (B, 256, D/4, H/4, W/4)

        # Block 1: (B, 64, D/4, H/4, W/4) -> (B, 64, D/4, H/4, W/4)
        # conv1 (1×1×1): 64 → 64
        self.layer1_block1_conv1 = nn.Conv3d(64, 64, kernel_size=1, bias=False)
        self.layer1_block1_bn1 = nn.BatchNorm3d(64)
        # conv2 (3×3×3): 64 → 64
        self.layer1_block1_conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn2 = nn.BatchNorm3d(64)
        # conv3 (1×1×1): 64 → 256
        self.layer1_block1_conv3 = nn.Conv3d(64, 256, kernel_size=1, bias=False)
        self.layer1_block1_bn3 = nn.BatchNorm3d(256)
        # residual downsample: 64 → 256
        self.layer1_block1_downsample_conv = nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False)
        self.layer1_block1_downsample_bn = nn.BatchNorm3d(256)

        # Block 2-3: (B, 256, D/4, H/4, W/4) -> (B, 256, D/4, H/4, W/4)
        # conv1 (1×1×1): 256 → 64
        self.layer1_block2_conv1 = nn.Conv3d(256, 64, kernel_size=1, bias=False)
        self.layer1_block2_bn1 = nn.BatchNorm3d(64)
        # conv2 (3×3×3): 64 → 64
        self.layer1_block2_conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block2_bn2 = nn.BatchNorm3d(64)
        # conv3 (1×1×1): 64 → 256
        self.layer1_block2_conv3 = nn.Conv3d(64, 256, kernel_size=1, bias=False)
        self.layer1_block2_bn3 = nn.BatchNorm3d(256)
        
        ## Layer 2: (B, 256, D/4, H/4, W/4) -> (B, 512, D/8, H/8, W/8)

        # Block 1: (B, 256, D/4, H/4, W/4) -> (B, 256, D/4, H/4, W/4)
        # conv1 (1×1×1): 256 → 128
        self.layer2_block1_conv1 = nn.Conv3d(256, 128, kernel_size=1, bias=False)
        self.layer2_block1_bn1 = nn.BatchNorm3d(128)
        # conv2 (3×3×3): 128 → 128, stride=2
        self.layer2_block1_conv2 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_block1_bn2 = nn.BatchNorm3d(128)
        # conv3 (1×1×1): 128 → 512
        self.layer2_block1_conv3 = nn.Conv3d(128, 512, kernel_size=1, bias=False)
        self.layer2_block1_bn3 = nn.BatchNorm3d(512)
        # residual downsample: 256 → 512, stride=2
        self.layer2_block1_downsample_conv = nn.Conv3d(256, 512, kernel_size=1, stride=2, bias=False)
        self.layer2_block1_downsample_bn = nn.BatchNorm3d(512)

        # Block 2-4: (B, 512, D/8, H/8, W/8) -> (B, 512, D/8, H/8, W/8)
        # conv1 (1×1×1): 512 → 128
        self.layer2_block2_conv1 = nn.Conv3d(512, 128, kernel_size=1, bias=False)
        self.layer2_block2_bn1 = nn.BatchNorm3d(128)
        # conv2 (3×3×3): 128 → 128, stride=1
        self.layer2_block2_conv2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block2_bn2 = nn.BatchNorm3d(128)
        # conv3 (1×1×1): 128 → 512
        self.layer2_block2_conv3 = nn.Conv3d(128, 512, kernel_size=1, bias=False)
        self.layer2_block2_bn3 = nn.BatchNorm3d(512)

        ## Layer 3: (B, 512, D/8, H/8, W/8) -> (B, 1024, D/16, H/16, W/16)

        # Block 1: (B, 512, D/8, H/8, W/8) -> (B, 1024, D/16, H/16, W/16)
        # conv1 (1×1×1): 512 → 256
        self.layer3_block1_conv1 = nn.Conv3d(512, 256, kernel_size=1, bias=False)
        self.layer3_block1_bn1 = nn.BatchNorm3d(256)
        # conv2 (3×3×3): 256 → 256, stride=2
        self.layer3_block1_conv2 = nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_block1_bn2 = nn.BatchNorm3d(256)
        # conv3 (1×1×1): 256 → 1024
        self.layer3_block1_conv3 = nn.Conv3d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block1_bn3 = nn.BatchNorm3d(1024)
        # residual downsample: 512 → 1024, stride=2
        self.layer3_block1_downsample_conv = nn.Conv3d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.layer3_block1_downsample_bn = nn.BatchNorm3d(1024)

        # Block 2-6: (B, 1024, D/16, H/16, W/16) -> (B, 1024, D/16, H/16, W/16)
        # conv1 (1×1×1): 1024 → 256
        self.layer3_block2_conv1 = nn.Conv3d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block2_bn1 = nn.BatchNorm3d(256)
        # conv2 (3×3×3): 256 → 256, stride=1
        self.layer3_block2_conv2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block2_bn2 = nn.BatchNorm3d(256)
        # conv3 (1×1×1): 256 → 1024
        self.layer3_block2_conv3 = nn.Conv3d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block2_bn3 = nn.BatchNorm3d(1024)

        ## Layer 4: (B, 1024, D/16, H/16, W/16) -> (B, 2048, D/32, H/32, W/32)

        # Block 1: (B, 1024, D/16, H/16, W/16) -> (B, 2048, D/32, H/32, W/32)
        # conv1 (1×1×1): 1024 → 512
        self.layer4_block1_conv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)
        self.layer4_block1_bn1 = nn.BatchNorm3d(512)
        # conv2 (3×3×3): 512 → 512, stride=2
        self.layer4_block1_conv2 = nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_block1_bn2 = nn.BatchNorm3d(512)
        # conv3 (1×1×1): 512 → 2048
        self.layer4_block1_conv3 = nn.Conv3d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block1_bn3 = nn.BatchNorm3d(2048)
        # residual downsample: 1024 → 2048, stride=2
        self.layer4_block1_downsample_conv = nn.Conv3d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.layer4_block1_downsample_bn = nn.BatchNorm3d(2048)

        # Block 2-3: (B, 2048, D/32, H/32, W/32) -> (B, 2048, D/32, H/32, W/32)
        # conv1 (1×1×1): 2048 → 512
        self.layer4_block2_conv1 = nn.Conv3d(2048, 512, kernel_size=1, bias=False)
        self.layer4_block2_bn1 = nn.BatchNorm3d(512)
        # conv2 (3×3×3): 512 → 512, stride=1
        self.layer4_block2_conv2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block2_bn2 = nn.BatchNorm3d(512)
        # conv3 (1×1×1): 512 → 2048
        self.layer4_block2_conv3 = nn.Conv3d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block2_bn3 = nn.BatchNorm3d(2048)

        # Global average pooling: (B, 2048, D/32, H/32, W/32) -> (B, 2048, 1, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Flatten: (B, 2048, 1, 1, 1) -> (B, 2048)
        self.flatten = nn.Flatten()

        # Linear: (B, 2048) -> (B, 1)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 3, depth, height, width)
            
        Returns:
            Logits tensor of shape (batch, 1) - apply sigmoid for probability
        """
        # Initial convolution: (B, 3, D, H, W) -> (B, 64, D/2, H/2, W/2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # MaxPooling: (B, 64, D/2, H/2, W/2) -> (B, 64, D/4, H/4, W/4)
        x = self.maxpool(x)

        # Layer 1: (B, 64, D/4, H/4, W/4) -> (B, 256, D/4, H/4, W/4)

        # Block 1
        identity = self.layer1_block1_downsample_bn(self.layer1_block1_downsample_conv(x))
        out = self.layer1_block1_conv1(x)
        out = self.layer1_block1_bn1(out)
        out = self.relu(out)
        out = self.layer1_block1_conv2(out)
        out = self.layer1_block1_bn2(out)
        out = self.relu(out)
        out = self.layer1_block1_conv3(out)
        out = self.layer1_block1_bn3(out)
        out = out + identity
        x = self.relu(out)
        
        # Block 2-3
        for _ in range(2):
            identity = x
            out = self.layer1_block2_conv1(x)
            out = self.layer1_block2_bn1(out)
            out = self.relu(out)
            out = self.layer1_block2_conv2(out)
            out = self.layer1_block2_bn2(out)
            out = self.relu(out)
            out = self.layer1_block2_conv3(out)
            out = self.layer1_block2_bn3(out)
            out = out + identity
            x = self.relu(out)

        # Layer 2: (B, 256, D/4, H/4, W/4) -> (B, 512, D/8, H/8, W/8)

        # Block 1
        identity = self.layer2_block1_downsample_bn(self.layer2_block1_downsample_conv(x))
        out = self.layer2_block1_conv1(x)
        out = self.layer2_block1_bn1(out)
        out = self.relu(out)
        out = self.layer2_block1_conv2(out)
        out = self.layer2_block1_bn2(out)
        out = self.relu(out)
        out = self.layer2_block1_conv3(out)
        out = self.layer2_block1_bn3(out)
        out = out + identity
        x = self.relu(out)
        
        # Block 2-4
        for _ in range(3):
            identity = x
            out = self.layer2_block2_conv1(x)
            out = self.layer2_block2_bn1(out)
            out = self.relu(out)
            out = self.layer2_block2_conv2(out)
            out = self.layer2_block2_bn2(out)
            out = self.relu(out)
            out = self.layer2_block2_conv3(out)
            out = self.layer2_block2_bn3(out)
            out = out + identity
            x = self.relu(out)

        # Layer 3: (B, 512, D/8, H/8, W/8) -> (B, 1024, D/16, H/16, W/16)

        # Block 1
        identity = self.layer3_block1_downsample_bn(self.layer3_block1_downsample_conv(x))
        out = self.layer3_block1_conv1(x)
        out = self.layer3_block1_bn1(out)
        out = self.relu(out)
        out = self.layer3_block1_conv2(out)
        out = self.layer3_block1_bn2(out)
        out = self.relu(out)
        out = self.layer3_block1_conv3(out)
        out = self.layer3_block1_bn3(out)
        out = out + identity
        x = self.relu(out)
        
        # Block 2-6
        for _ in range(5):
            identity = x
            out = self.layer3_block2_conv1(x)
            out = self.layer3_block2_bn1(out)
            out = self.relu(out)
            out = self.layer3_block2_conv2(out)
            out = self.layer3_block2_bn2(out)
            out = self.relu(out)
            out = self.layer3_block2_conv3(out)
            out = self.layer3_block2_bn3(out)
            out = out + identity
            x = self.relu(out)

        # Layer 4: (B, 1024, D/16, H/16, W/16) -> (B, 2048, D/32, H/32, W/32)

        # Block 1
        identity = self.layer4_block1_downsample_bn(self.layer4_block1_downsample_conv(x))
        out = self.layer4_block1_conv1(x)
        out = self.layer4_block1_bn1(out)
        out = self.relu(out)
        out = self.layer4_block1_conv2(out)
        out = self.layer4_block1_bn2(out)
        out = self.relu(out)
        out = self.layer4_block1_conv3(out)
        out = self.layer4_block1_bn3(out)
        out = out + identity
        x = self.relu(out)
        
        # Block 2-3
        for _ in range(2):
            identity = x
            out = self.layer4_block2_conv1(x)
            out = self.layer4_block2_bn1(out)
            out = self.relu(out)
            out = self.layer4_block2_conv2(out)
            out = self.layer4_block2_bn2(out)
            out = self.relu(out)
            out = self.layer4_block2_conv3(out)
            out = self.layer4_block2_bn3(out)
            out = out + identity
            x = self.relu(out)

        # Global average pooling: (B, 2048, D/32, H/32, W/32) -> (B, 2048, 1, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (B, 2048, 1, 1, 1) -> (B, 2048)
        x = self.flatten(x)
        
        # Linear: (B, 2048) -> (B, 1)
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
