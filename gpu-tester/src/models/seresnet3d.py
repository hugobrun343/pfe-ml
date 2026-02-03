"""
SE-ResNet3D (Squeeze-and-Excitation ResNet) for 3D medical image classification
Adds channel attention mechanism to ResNet blocks
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock3D, self).__init__()
        
        reduced_channels = max(channels // reduction, 1)
        
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1, 1)
        
        # Scale: multiply input with channel weights
        return x * y.expand_as(x)


class SEBasicBlock3D(nn.Module):
    """SE-ResNet basic block with Squeeze-and-Excitation"""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16
    ):
        super(SEBasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # SE block
        self.se = SEBlock3D(out_channels, reduction)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE block
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class SEBottleneck3D(nn.Module):
    """SE-ResNet bottleneck block with Squeeze-and-Excitation"""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16
    ):
        super(SEBottleneck3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        
        # SE block
        self.se = SEBlock3D(out_channels * self.expansion, reduction)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE block
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class SEResNet3D(nn.Module):
    """SE-ResNet3D architecture"""
    
    def __init__(
        self,
        block,
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 2,
        initial_channels: int = 64,
        reduction: int = 16
    ):
        super(SEResNet3D, self).__init__()
        
        self.in_channels = initial_channels
        self.reduction = reduction
        
        # Initial convolution layer
        self.conv1 = nn.Conv3d(
            in_channels, initial_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        block,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample, self.reduction)
        )
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, reduction=self.reduction)
            )
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# SE-ResNet3D configurations (5 variants)
def get_seresnet3d_18(in_channels: int = 1, num_classes: int = 2, reduction: int = 16) -> SEResNet3D:
    """SE-ResNet3D-18 - Small variant with SE attention"""
    return SEResNet3D(
        SEBasicBlock3D, [2, 2, 2, 2],
        in_channels, num_classes,
        reduction=reduction
    )


def get_seresnet3d_34(in_channels: int = 1, num_classes: int = 2, reduction: int = 16) -> SEResNet3D:
    """SE-ResNet3D-34 - Medium variant with SE attention"""
    return SEResNet3D(
        SEBasicBlock3D, [3, 4, 6, 3],
        in_channels, num_classes,
        reduction=reduction
    )


def get_seresnet3d_50(in_channels: int = 1, num_classes: int = 2, reduction: int = 16) -> SEResNet3D:
    """SE-ResNet3D-50 - Large variant with SE attention"""
    return SEResNet3D(
        SEBottleneck3D, [3, 4, 6, 3],
        in_channels, num_classes,
        reduction=reduction
    )


def get_seresnet3d_101(in_channels: int = 1, num_classes: int = 2, reduction: int = 16) -> SEResNet3D:
    """SE-ResNet3D-101 - XLarge variant with SE attention"""
    return SEResNet3D(
        SEBottleneck3D, [3, 4, 23, 3],
        in_channels, num_classes,
        reduction=reduction
    )


def get_seresnet3d_152(in_channels: int = 1, num_classes: int = 2, reduction: int = 16) -> SEResNet3D:
    """SE-ResNet3D-152 - XXLarge variant with SE attention"""
    return SEResNet3D(
        SEBottleneck3D, [3, 8, 36, 3],
        in_channels, num_classes,
        reduction=reduction
    )
