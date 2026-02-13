"""
ConvNeXt3D-Large for 3D medical image classification

Input: (B, 3, 32, 256, 256) - batch, channels, depth, height, width

Architecture: ConvNeXt Large adapted for 3D volumetric data
    - Stem: Patchify with 4×4×4 non-overlapping convolution
    - 4 Stages with ConvNeXt blocks
    - Downsampling between stages with 2×2×2 strided convolution
    - Global Average Pooling + Linear head

ConvNeXt-Large Configuration:
    - Dims: [192, 384, 768, 1536]
    - Depths: [3, 3, 27, 3] (36 total blocks)
    - Drop path rate: 0.1 (linearly increasing)

ConvNeXt Block structure (inverted bottleneck):
    - Depthwise Conv 7×7×7 (spatial mixing, groups=C)
    - LayerNorm
    - Pointwise Conv 1×1 (channel expansion: C -> 4C)
    - GELU activation
    - Pointwise Conv 1×1 (channel reduction: 4C -> C)
    - Layer Scale (learnable gamma, init=1e-6)
    - Drop Path (stochastic depth)
    - Residual connection

Total parameters: ~198M

Dimensions for input (B, 3, 32, 256, 256):
    Stem:    (B, 3, 32, 256, 256)   -> (B, 192, 8, 64, 64)
    Stage 1: (B, 192, 8, 64, 64)    -> (B, 192, 8, 64, 64)    [3 blocks]
    Down 1:  (B, 192, 8, 64, 64)    -> (B, 384, 4, 32, 32)
    Stage 2: (B, 384, 4, 32, 32)    -> (B, 384, 4, 32, 32)    [3 blocks]
    Down 2:  (B, 384, 4, 32, 32)    -> (B, 768, 2, 16, 16)
    Stage 3: (B, 768, 2, 16, 16)    -> (B, 768, 2, 16, 16)    [27 blocks]
    Down 3:  (B, 768, 2, 16, 16)    -> (B, 1536, 1, 8, 8)
    Stage 4: (B, 1536, 1, 8, 8)     -> (B, 1536, 1, 8, 8)     [3 blocks]
    Head:    (B, 1536, 1, 8, 8)     -> (B, num_classes)
"""

import torch
import torch.nn as nn


class ConvNeXt3DLarge(nn.Module):
    """ConvNeXt3D-Large for binary classification
    
    Configuration:
        - Dims: [192, 384, 768, 1536]
        - Depths: [3, 3, 27, 3] = 36 blocks
        - Drop path: 0.0 -> 0.1 (linear)
        - Total params: ~198M
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1, drop_path_rate: float = 0.1):
        """
        Initialize ConvNeXt3D-Large
        
        Args:
            in_channels: Number of input channels (default: 3)
            num_classes: Number of output classes (default: 1 for binary)
            drop_path_rate: Maximum drop path rate (default: 0.1)
        """
        super(ConvNeXt3DLarge, self).__init__()
        
        # ============================================================================
        # CONFIGURATION
        # ============================================================================
        self.dims = [192, 384, 768, 1536]
        self.depths = [3, 3, 27, 3]  # 36 total blocks
        
        # Calculate drop path rates (linearly increasing from 0 to drop_path_rate)
        total_blocks = sum(self.depths)  # 36
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # ============================================================================
        # STEM: (B, 3, 32, 256, 256) -> (B, 192, 8, 64, 64)
        # Patchify with 4×4×4 non-overlapping convolution (like ViT patch embedding)
        # ============================================================================
        
        # Conv3d (4×4×4), stride=4: 3 → 192
        # (B, 3, 32, 256, 256) -> (B, 192, 8, 64, 64)
        self.stem_conv = nn.Conv3d(in_channels, self.dims[0], kernel_size=4, stride=4)
        # LayerNorm: 192 (applied in channel-last format)
        self.stem_norm = nn.LayerNorm(self.dims[0])
        
        # ============================================================================
        # STAGE 1: (B, 192, 8, 64, 64) -> (B, 192, 8, 64, 64) - 3 blocks
        # Each block: DWConv 7×7×7 -> LN -> Linear 192→768 -> GELU -> Linear 768→192
        # ============================================================================
        cur = 0
        self.stage1_blocks = nn.ModuleList()
        for i in range(self.depths[0]):
            # Block i: drop_path_rate increases linearly
            self.stage1_blocks.append(ConvNeXtBlock3D(
                dim=self.dims[0],  # 192
                drop_path_rate=dp_rates[cur + i]
            ))
        cur += self.depths[0]  # cur = 3
        
        # ============================================================================
        # DOWNSAMPLE 1->2: (B, 192, 8, 64, 64) -> (B, 384, 4, 32, 32)
        # LayerNorm + Conv3d 2×2×2, stride=2
        # ============================================================================
        # LayerNorm: 192
        self.down1_norm = nn.LayerNorm(self.dims[0])
        # Conv3d (2×2×2), stride=2: 192 → 384
        self.down1_conv = nn.Conv3d(self.dims[0], self.dims[1], kernel_size=2, stride=2)
        
        # ============================================================================
        # STAGE 2: (B, 384, 4, 32, 32) -> (B, 384, 4, 32, 32) - 3 blocks
        # Each block: DWConv 7×7×7 -> LN -> Linear 384→1536 -> GELU -> Linear 1536→384
        # ============================================================================
        self.stage2_blocks = nn.ModuleList()
        for i in range(self.depths[1]):
            self.stage2_blocks.append(ConvNeXtBlock3D(
                dim=self.dims[1],  # 384
                drop_path_rate=dp_rates[cur + i]
            ))
        cur += self.depths[1]  # cur = 6
        
        # ============================================================================
        # DOWNSAMPLE 2->3: (B, 384, 4, 32, 32) -> (B, 768, 2, 16, 16)
        # ============================================================================
        # LayerNorm: 384
        self.down2_norm = nn.LayerNorm(self.dims[1])
        # Conv3d (2×2×2), stride=2: 384 → 768
        self.down2_conv = nn.Conv3d(self.dims[1], self.dims[2], kernel_size=2, stride=2)
        
        # ============================================================================
        # STAGE 3: (B, 768, 2, 16, 16) -> (B, 768, 2, 16, 16) - 27 blocks (main stage)
        # Each block: DWConv 7×7×7 -> LN -> Linear 768→3072 -> GELU -> Linear 3072→768
        # ============================================================================
        self.stage3_blocks = nn.ModuleList()
        for i in range(self.depths[2]):
            self.stage3_blocks.append(ConvNeXtBlock3D(
                dim=self.dims[2],  # 768
                drop_path_rate=dp_rates[cur + i]
            ))
        cur += self.depths[2]  # cur = 33
        
        # ============================================================================
        # DOWNSAMPLE 3->4: (B, 768, 2, 16, 16) -> (B, 1536, 1, 8, 8)
        # ============================================================================
        # LayerNorm: 768
        self.down3_norm = nn.LayerNorm(self.dims[2])
        # Conv3d (2×2×2), stride=2: 768 → 1536
        self.down3_conv = nn.Conv3d(self.dims[2], self.dims[3], kernel_size=2, stride=2)
        
        # ============================================================================
        # STAGE 4: (B, 1536, 1, 8, 8) -> (B, 1536, 1, 8, 8) - 3 blocks
        # Each block: DWConv 7×7×7 -> LN -> Linear 1536→6144 -> GELU -> Linear 6144→1536
        # ============================================================================
        self.stage4_blocks = nn.ModuleList()
        for i in range(self.depths[3]):
            self.stage4_blocks.append(ConvNeXtBlock3D(
                dim=self.dims[3],  # 1536
                drop_path_rate=dp_rates[cur + i]
            ))
        
        # ============================================================================
        # HEAD: (B, 1536, 1, 8, 8) -> (B, num_classes)
        # ============================================================================
        
        # Global Average Pooling: (B, 1536, 1, 8, 8) -> (B, 1536, 1, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        # Final LayerNorm: 1536
        self.final_norm = nn.LayerNorm(self.dims[3])
        
        # Classification head: 1536 → num_classes
        self.head = nn.Linear(self.dims[3], num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 3, 32, 256, 256)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # ============================== STEM ==============================
        # Conv3d (4×4×4), stride=4: (B, 3, 32, 256, 256) -> (B, 192, 8, 64, 64)
        x = self.stem_conv(x)
        # LayerNorm in channel-last format
        x = x.permute(0, 2, 3, 4, 1)  # (B, 192, 8, 64, 64) -> (B, 8, 64, 64, 192)
        x = self.stem_norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, 8, 64, 64, 192) -> (B, 192, 8, 64, 64)
        
        # ============================ STAGE 1 =============================
        # (B, 192, 8, 64, 64) -> (B, 192, 8, 64, 64) - 3 blocks
        for block in self.stage1_blocks:
            x = block(x)
        
        # ========================== DOWNSAMPLE 1 ==========================
        # LayerNorm + Conv3d: (B, 192, 8, 64, 64) -> (B, 384, 4, 32, 32)
        x = x.permute(0, 2, 3, 4, 1)  # -> (B, D, H, W, C)
        x = self.down1_norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # -> (B, C, D, H, W)
        x = self.down1_conv(x)        # 192 → 384, stride=2
        
        # ============================ STAGE 2 =============================
        # (B, 384, 4, 32, 32) -> (B, 384, 4, 32, 32) - 3 blocks
        for block in self.stage2_blocks:
            x = block(x)
        
        # ========================== DOWNSAMPLE 2 ==========================
        # LayerNorm + Conv3d: (B, 384, 4, 32, 32) -> (B, 768, 2, 16, 16)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.down2_norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.down2_conv(x)        # 384 → 768, stride=2
        
        # ============================ STAGE 3 =============================
        # (B, 768, 2, 16, 16) -> (B, 768, 2, 16, 16) - 27 blocks (main stage)
        for block in self.stage3_blocks:
            x = block(x)
        
        # ========================== DOWNSAMPLE 3 ==========================
        # LayerNorm + Conv3d: (B, 768, 2, 16, 16) -> (B, 1536, 1, 8, 8)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.down3_norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.down3_conv(x)        # 768 → 1536, stride=2
        
        # ============================ STAGE 4 =============================
        # (B, 1536, 1, 8, 8) -> (B, 1536, 1, 8, 8) - 3 blocks
        for block in self.stage4_blocks:
            x = block(x)
        
        # ============================== HEAD ==============================
        # Global Average Pooling: (B, 1536, 1, 8, 8) -> (B, 1536, 1, 1, 1)
        x = self.avgpool(x)
        # Flatten: (B, 1536, 1, 1, 1) -> (B, 1536)
        x = x.flatten(1)
        # LayerNorm: (B, 1536) -> (B, 1536)
        x = self.final_norm(x)
        # Classification: (B, 1536) -> (B, num_classes)
        x = self.head(x)
        
        return x


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt Block for 3D (inverted bottleneck with depthwise conv)
    
    Structure:
        x -> DWConv 7×7×7 -> LayerNorm -> Linear (C→4C) -> GELU -> Linear (4C→C) -> LayerScale -> DropPath -> + x
    
    For dim=192:  DWConv 192→192 -> LN -> Linear 192→768 -> GELU -> Linear 768→192
    For dim=384:  DWConv 384→384 -> LN -> Linear 384→1536 -> GELU -> Linear 1536→384
    For dim=768:  DWConv 768→768 -> LN -> Linear 768→3072 -> GELU -> Linear 3072→768
    For dim=1536: DWConv 1536→1536 -> LN -> Linear 1536→6144 -> GELU -> Linear 6144→1536
    
    Args:
        dim: Number of input/output channels
        drop_path_rate: Drop path probability for stochastic depth
    """
    
    def __init__(self, dim: int, drop_path_rate: float = 0.0):
        super(ConvNeXtBlock3D, self).__init__()
        
        # Depthwise convolution (7×7×7), groups=dim: C → C
        # Each channel is convolved independently (spatial mixing)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm: C (applied in channel-last format)
        self.norm = nn.LayerNorm(dim)
        
        # Pointwise convolutions (channel mixing via Linear layers)
        # Inverted bottleneck: expand 4x then contract
        # Linear: C → 4C
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # GELU activation
        self.act = nn.GELU()
        # Linear: 4C → C
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale: learnable per-channel scaling (init=1e-6 for stability)
        # gamma shape: (C,)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        
        # Drop path rate for stochastic depth
        self.drop_path_rate = drop_path_rate
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Output tensor (B, C, D, H, W)
        """
        shortcut = x
        
        # Depthwise convolution (7×7×7): C → C (spatial mixing)
        # (B, C, D, H, W) -> (B, C, D, H, W)
        x = self.dwconv(x)
        
        # Convert to channel-last for LayerNorm and Linear
        # (B, C, D, H, W) -> (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)
        
        # LayerNorm: (B, D, H, W, C) -> (B, D, H, W, C)
        x = self.norm(x)
        # Linear: C → 4C
        x = self.pwconv1(x)
        # GELU activation
        x = self.act(x)
        # Linear: 4C → C
        x = self.pwconv2(x)
        
        # Layer Scale: element-wise multiply by gamma
        # (B, D, H, W, C) * (C,) -> (B, D, H, W, C)
        x = self.gamma * x
        
        # Convert back to channel-first
        # (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Drop Path (stochastic depth) during training
        if self.drop_path_rate > 0.0 and self.training:
            keep_prob = 1.0 - self.drop_path_rate
            # Create binary mask: (B, 1, 1, 1, 1)
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # Binary mask
            x = x.div(keep_prob) * random_tensor
        
        # Residual connection
        x = shortcut + x
        
        return x


def create_model(in_channels: int = 3, num_classes: int = 1) -> ConvNeXt3DLarge:
    """
    Factory function to create ConvNeXt3D-Large model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1)
        
    Returns:
        ConvNeXt3DLarge model instance
    """
    return ConvNeXt3DLarge(in_channels=in_channels, num_classes=num_classes)
