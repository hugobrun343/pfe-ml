"""
ViT3D-Large (Vision Transformer) for 3D medical image classification

Input: (B, 3, 32, 256, 256) - batch, channels, depth, height, width

Architecture: Vision Transformer Large adapted for 3D volumetric data
    - Patch Embedding: Conv3d with patch_size stride (non-overlapping patches)
    - Class Token: Learnable [CLS] token prepended to patch sequence
    - Position Embedding: Learnable absolute position embeddings
    - Transformer Encoder: 24 layers of Multi-Head Self-Attention + MLP
    - Classification Head: LayerNorm + Linear on [CLS] token

ViT-Large Configuration:
    - Patch size: (4, 16, 16) for (D, H, W)
    - Hidden dim: 1024
    - MLP dim: 4096 (4x hidden)
    - Attention heads: 16
    - Transformer layers: 24
    - Head dim: 1024 / 16 = 64

Transformer Block structure:
    - LayerNorm -> Multi-Head Self-Attention -> Residual
    - LayerNorm -> MLP (FC -> GELU -> FC) -> Residual

Multi-Head Self-Attention:
    - Q, K, V projection: (B, N, 1024) -> (B, N, 1024*3) -> split
    - Attention: softmax(Q @ K^T / sqrt(d)) @ V
    - Output projection: (B, N, 1024) -> (B, N, 1024)

Total parameters: ~304M

Dimensions for input (B, 3, 32, 256, 256):
    Patch Embed: (B, 3, 32, 256, 256) -> (B, 1024, 8, 16, 16) -> (B, 2048, 1024)
    + CLS Token: (B, 2048, 1024) -> (B, 2049, 1024)
    + Pos Embed: (B, 2049, 1024) -> (B, 2049, 1024)
    Transformer: (B, 2049, 1024) -> (B, 2049, 1024) [x24 layers]
    Head:        (B, 2049, 1024) -> (B, 1024) -> (B, num_classes)
"""

import torch
import torch.nn as nn


class ViT3DLarge(nn.Module):
    """ViT3D-Large for binary classification
    
    Configuration:
        - Patch size: (4, 16, 16) -> 2048 patches
        - Hidden dim: 1024, MLP dim: 4096
        - Heads: 16, Layers: 24
        - Total params: ~304M
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        """
        Initialize ViT3D-Large
        
        Args:
            in_channels: Number of input channels (default: 3)
            num_classes: Number of output classes (default: 1 for binary)
        """
        super(ViT3DLarge, self).__init__()
        
        # ============================================================================
        # CONFIGURATION
        # ============================================================================
        self.patch_size = (4, 16, 16)  # (D, H, W)
        self.hidden_dim = 1024
        self.mlp_dim = 4096            # 4x hidden_dim
        self.num_heads = 16
        self.num_layers = 24
        self.head_dim = self.hidden_dim // self.num_heads  # 64
        
        # Number of patches: (32/4) * (256/16) * (256/16) = 8 * 16 * 16 = 2048
        self.num_patches = 8 * 16 * 16
        
        # ============================================================================
        # PATCH EMBEDDING: (B, 3, 32, 256, 256) -> (B, 2048, 1024)
        # ============================================================================
        
        # Conv3d (4×16×16), stride=(4,16,16): 3 → 1024
        # (B, 3, 32, 256, 256) -> (B, 1024, 8, 16, 16) -> flatten -> (B, 2048, 1024)
        self.patch_embed = nn.Conv3d(
            in_channels, self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # ============================================================================
        # CLASS TOKEN & POSITION EMBEDDINGS
        # ============================================================================
        
        # Learnable [CLS] token: (1, 1, 1024)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        # Learnable position embeddings: (1, 2049, 1024) for 2048 patches + 1 CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_dim))
        
        # Dropout after position embedding: p=0.1
        self.pos_dropout = nn.Dropout(p=0.1)
        
        # ============================================================================
        # TRANSFORMER ENCODER: 24 layers (created dynamically)
        # Each layer: LayerNorm -> MHSA -> Residual -> LayerNorm -> MLP -> Residual
        # ============================================================================
        
        for i in range(1, 25):
            # LayerNorm: 1024
            setattr(self, f'layer{i}_norm1', nn.LayerNorm(self.hidden_dim))
            # QKV Linear: 1024 → 3072 (1024 × 3 for Q, K, V)
            setattr(self, f'layer{i}_attn_qkv', nn.Linear(self.hidden_dim, self.hidden_dim * 3))
            # Output projection: 1024 → 1024
            setattr(self, f'layer{i}_attn_proj', nn.Linear(self.hidden_dim, self.hidden_dim))
            # Attention dropout: p=0.1
            setattr(self, f'layer{i}_attn_dropout', nn.Dropout(p=0.1))
            # LayerNorm: 1024
            setattr(self, f'layer{i}_norm2', nn.LayerNorm(self.hidden_dim))
            # MLP fc1: 1024 → 4096
            setattr(self, f'layer{i}_mlp_fc1', nn.Linear(self.hidden_dim, self.mlp_dim))
            # MLP fc2: 4096 → 1024
            setattr(self, f'layer{i}_mlp_fc2', nn.Linear(self.mlp_dim, self.hidden_dim))
            # MLP dropout: p=0.1
            setattr(self, f'layer{i}_mlp_dropout', nn.Dropout(p=0.1))
        
        # ============================================================================
        # HEAD: (B, 1024) -> (B, num_classes)
        # ============================================================================
        
        # Final LayerNorm: 1024
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
        # Classification head: 1024 → num_classes
        self.head = nn.Linear(self.hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings and class token with truncated normal"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def _attention(self, x, qkv_layer, proj_layer, dropout_layer):
        """
        Multi-Head Self-Attention
        
        Structure:
            1. QKV projection: (B, N, 1024) -> (B, N, 3072) -> split to Q, K, V
            2. Reshape: (B, N, 1024) -> (B, 16 heads, N, 64)
            3. Attention: softmax(Q @ K^T / sqrt(64)) @ V
            4. Merge heads: (B, 16, N, 64) -> (B, N, 1024)
            5. Output projection: (B, N, 1024) -> (B, N, 1024)
        
        Args:
            x: Input tensor (B, N, 1024)
            qkv_layer: Linear layer for Q, K, V projection
            proj_layer: Linear layer for output projection
            dropout_layer: Dropout for attention weights
            
        Returns:
            Output tensor (B, N, 1024)
        """
        B, N, C = x.shape
        
        # QKV projection: (B, N, 1024) -> (B, N, 3072) -> (B, N, 3, 16, 64)
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # Permute: (B, N, 3, 16, 64) -> (3, B, 16, N, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, 16, N, 64)
        
        # Scaled dot-product attention
        # attn = softmax(Q @ K^T / sqrt(d_k)) where d_k = 64
        scale = (C // self.num_heads) ** -0.5  # 1/8
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, 16, N, N)
        attn = attn.softmax(dim=-1)
        attn = dropout_layer(attn)
        
        # Apply attention to values: (B, 16, N, N) @ (B, 16, N, 64) -> (B, 16, N, 64)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, 1024)
        
        # Output projection: (B, N, 1024) -> (B, N, 1024)
        x = proj_layer(x)
        
        return x
    
    def _mlp(self, x, fc1, fc2, dropout):
        """
        MLP block with GELU activation
        
        Structure: Linear (1024 → 4096) -> GELU -> Dropout -> Linear (4096 → 1024) -> Dropout
        
        Args:
            x: Input tensor (B, N, 1024)
            fc1: First linear layer (1024 → 4096)
            fc2: Second linear layer (4096 → 1024)
            dropout: Dropout layer
            
        Returns:
            Output tensor (B, N, 1024)
        """
        x = fc1(x)                          # (B, N, 1024) -> (B, N, 4096)
        x = torch.nn.functional.gelu(x)     # GELU activation
        x = dropout(x)
        x = fc2(x)                          # (B, N, 4096) -> (B, N, 1024)
        x = dropout(x)
        return x
    
    def _transformer_block(self, x, layer_idx):
        """
        Single Transformer encoder block (Pre-LN architecture)
        
        Structure:
            x = x + MHSA(LayerNorm(x))
            x = x + MLP(LayerNorm(x))
            
        Args:
            x: Input tensor (B, N, 1024)
            layer_idx: Layer index (1-24)
            
        Returns:
            Output tensor (B, N, 1024)
        """
        # Get layer components by index
        norm1 = getattr(self, f'layer{layer_idx}_norm1')
        qkv = getattr(self, f'layer{layer_idx}_attn_qkv')
        proj = getattr(self, f'layer{layer_idx}_attn_proj')
        attn_drop = getattr(self, f'layer{layer_idx}_attn_dropout')
        norm2 = getattr(self, f'layer{layer_idx}_norm2')
        fc1 = getattr(self, f'layer{layer_idx}_mlp_fc1')
        fc2 = getattr(self, f'layer{layer_idx}_mlp_fc2')
        mlp_drop = getattr(self, f'layer{layer_idx}_mlp_dropout')
        
        # Self-attention with pre-norm and residual
        x = x + self._attention(norm1(x), qkv, proj, attn_drop)
        
        # MLP with pre-norm and residual
        x = x + self._mlp(norm2(x), fc1, fc2, mlp_drop)
        
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 3, 32, 256, 256)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        B = x.shape[0]
        
        # ========================= PATCH EMBEDDING ==========================
        # Conv3d: (B, 3, 32, 256, 256) -> (B, 1024, 8, 16, 16)
        # Flatten: (B, 1024, 8, 16, 16) -> (B, 1024, 2048) -> (B, 2048, 1024)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # =========================== CLASS TOKEN ============================
        # Prepend [CLS] token: (B, 2048, 1024) -> (B, 2049, 1024)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, 1024) -> (B, 1, 1024)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # ======================= POSITION EMBEDDING =========================
        # Add learnable position embeddings: (B, 2049, 1024) + (1, 2049, 1024)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # ====================== TRANSFORMER ENCODER =========================
        # 24 layers: (B, 2049, 1024) -> (B, 2049, 1024)
        for i in range(1, 25):
            x = self._transformer_block(x, i)
        
        # ============================ HEAD ==================================
        # LayerNorm: (B, 2049, 1024) -> (B, 2049, 1024)
        x = self.final_norm(x)
        # Extract [CLS] token: (B, 2049, 1024) -> (B, 1024)
        x = x[:, 0]
        # Classification: (B, 1024) -> (B, num_classes)
        x = self.head(x)
        
        return x


def create_model(in_channels: int = 3, num_classes: int = 1) -> ViT3DLarge:
    """
    Factory function to create ViT3D-Large model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1)
        
    Returns:
        ViT3DLarge model instance
    """
    return ViT3DLarge(in_channels=in_channels, num_classes=num_classes)
