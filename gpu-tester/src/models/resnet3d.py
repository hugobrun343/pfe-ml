"""
3D ResNet models for 3D medical image classification
Implements ResNet3D variants with different depths (10, 14, 18, 26, 34, 50, 101, 152)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class BasicBlock3D(nn.Module):
    """Basic 3D ResNet block"""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock3D, self).__init__()
        
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
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck3D(nn.Module):
    """Bottleneck 3D ResNet block for deeper networks"""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(Bottleneck3D, self).__init__()
        
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
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """3D ResNet architecture"""
    
    def __init__(
        self,
        block,
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 2,
        initial_channels: int = 64
    ):
        super(ResNet3D, self).__init__()
        
        self.in_channels = initial_channels
        
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
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
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


# ResNet3D factory functions (5 variants)
def get_resnet3d_10(in_channels: int = 1, num_classes: int = 2) -> ResNet3D:
    """ResNet3D-10 - Tiny variant"""
    return ResNet3D(BasicBlock3D, [1, 1, 1, 1], in_channels, num_classes)


def get_resnet3d_18(in_channels: int = 1, num_classes: int = 2) -> ResNet3D:
    """ResNet3D-18 - Small variant"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels, num_classes)


def get_resnet3d_34(in_channels: int = 1, num_classes: int = 2) -> ResNet3D:
    """ResNet3D-34 - Medium variant"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels, num_classes)


def get_resnet3d_50(in_channels: int = 1, num_classes: int = 2) -> ResNet3D:
    """ResNet3D-50 - Large variant with bottleneck"""
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels, num_classes)


def get_resnet3d_101(in_channels: int = 1, num_classes: int = 2) -> ResNet3D:
    """ResNet3D-101 - XLarge variant with bottleneck"""
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3], in_channels, num_classes)
