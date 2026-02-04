"""
Vision Transformer 3D (ViT3D) for 3D medical image classification
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super(PatchEmbed3D, self).__init__()
        self.patch_size = patch_size
        
        # Use 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention3D(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super(MultiHeadAttention3D, self).__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention3D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """Vision Transformer for 3D volumes"""
    
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0
    ):
        super(VisionTransformer3D, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (will be created dynamically based on input size)
        self.pos_embed = None
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def _get_pos_embed(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Get or create positional embedding"""
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches + 1:
            # Create new positional embedding
            pos_embed = torch.zeros(1, num_patches + 1, self.embed_dim, device=device)
            
            # Sinusoidal positional encoding
            position = torch.arange(0, num_patches + 1, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=device) *
                -(math.log(10000.0) / self.embed_dim)
            )
            
            pos_embed[0, :, 0::2] = torch.sin(position * div_term)
            pos_embed[0, :, 1::2] = torch.cos(position * div_term)
            
            self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        
        return self.pos_embed
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        pos_embed = self._get_pos_embed(x.shape[1] - 1, x.device)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification head
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take class token
        x = self.head(cls_token_final)
        
        return x


# ViT3D configurations
def get_vit3d_tiny(in_channels: int = 1, num_classes: int = 2, patch_size: int = 16) -> VisionTransformer3D:
    """ViT3D-Tiny: 5.7M params"""
    return VisionTransformer3D(
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0
    )


def get_vit3d_small(in_channels: int = 1, num_classes: int = 2, patch_size: int = 16) -> VisionTransformer3D:
    """ViT3D-Small: 22M params"""
    return VisionTransformer3D(
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0
    )


def get_vit3d_base(in_channels: int = 1, num_classes: int = 2, patch_size: int = 16) -> VisionTransformer3D:
    """ViT3D-Base: 86M params"""
    return VisionTransformer3D(
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.0
    )


def get_vit3d_large(in_channels: int = 1, num_classes: int = 2, patch_size: int = 16) -> VisionTransformer3D:
    """ViT3D-Large: 304M params"""
    return VisionTransformer3D(
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.0
    )
