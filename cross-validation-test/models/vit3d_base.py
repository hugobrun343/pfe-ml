"""
ViT3D-Base (Vision Transformer) for 3D medical image classification

Input: (B, 3, 32, 256, 256) - batch, channels, depth, height, width

Architecture: Vision Transformer adapted for 3D volumetric data
    - Patch Embedding: Conv3d with patch_size stride (non-overlapping patches)
    - Class Token: Learnable [CLS] token prepended to patch sequence
    - Position Embedding: Learnable absolute position embeddings
    - Transformer Encoder: 12 layers of Multi-Head Self-Attention + MLP
    - Classification Head: LayerNorm + Linear on [CLS] token

ViT-Base Configuration:
    - Patch size: (4, 16, 16) for (D, H, W)
    - Hidden dim: 768
    - MLP dim: 3072 (4x hidden)
    - Attention heads: 12
    - Transformer layers: 12
    - Head dim: 768 / 12 = 64

Transformer Block structure:
    - LayerNorm -> Multi-Head Self-Attention -> Residual
    - LayerNorm -> MLP (FC -> GELU -> FC) -> Residual

Multi-Head Self-Attention:
    - Q, K, V projection: (B, N, 768) -> (B, N, 768*3) -> split
    - Attention: softmax(Q @ K^T / sqrt(d)) @ V
    - Output projection: (B, N, 768) -> (B, N, 768)

Total parameters: ~86M

Dimensions for input (B, 3, 32, 256, 256):
    Patch Embed: (B, 3, 32, 256, 256) -> (B, 768, 8, 16, 16) -> (B, 2048, 768)
    + CLS Token: (B, 2048, 768) -> (B, 2049, 768)
    + Pos Embed: (B, 2049, 768) -> (B, 2049, 768)
    Transformer: (B, 2049, 768) -> (B, 2049, 768) [x12 layers]
    Head:        (B, 2049, 768) -> (B, 768) -> (B, num_classes)
"""

import torch
import torch.nn as nn


class ViT3DBase(nn.Module):
    """ViT3D-Base for binary classification
    
    Configuration:
        - Patch size: (4, 16, 16) -> 2048 patches
        - Hidden dim: 768, MLP dim: 3072
        - Heads: 12, Layers: 12
        - Total params: ~86M
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        """
        Initialize ViT3D-Base
        
        Args:
            in_channels: Number of input channels (default: 3)
            num_classes: Number of output classes (default: 1 for binary)
        """
        super(ViT3DBase, self).__init__()
        
        # ============================================================================
        # CONFIGURATION
        # ============================================================================
        self.patch_size = (4, 16, 16)  # (D, H, W)
        self.hidden_dim = 768
        self.mlp_dim = 3072            # 4x hidden_dim
        self.num_heads = 12
        self.num_layers = 12
        self.head_dim = self.hidden_dim // self.num_heads  # 64
        
        # Number of patches: (32/4) * (256/16) * (256/16) = 8 * 16 * 16 = 2048
        self.num_patches = 8 * 16 * 16
        
        # ============================================================================
        # PATCH EMBEDDING: (B, 3, 32, 256, 256) -> (B, 2048, 768)
        # ============================================================================
        
        # Conv3d (4×16×16), stride=(4,16,16): 3 → 768
        # (B, 3, 32, 256, 256) -> (B, 768, 8, 16, 16) -> flatten -> (B, 2048, 768)
        self.patch_embed = nn.Conv3d(
            in_channels, self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # ============================================================================
        # CLASS TOKEN & POSITION EMBEDDINGS
        # ============================================================================
        
        # Learnable [CLS] token: (1, 1, 768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        # Learnable position embeddings: (1, 2049, 768) for 2048 patches + 1 CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_dim))
        
        # Dropout after position embedding: p=0.1
        self.pos_dropout = nn.Dropout(p=0.1)
        
        # ============================================================================
        # TRANSFORMER ENCODER: 12 layers
        # Each layer: LayerNorm -> MHSA -> Residual -> LayerNorm -> MLP -> Residual
        # ============================================================================
        
        # ----------------------------------------------------------------------------
        # Layer 1
        # ----------------------------------------------------------------------------
        # LayerNorm: 768
        self.layer1_norm1 = nn.LayerNorm(self.hidden_dim)
        # QKV Linear: 768 → 2304 (768 × 3 for Q, K, V)
        self.layer1_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        # Output projection: 768 → 768
        self.layer1_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Attention dropout: p=0.1
        self.layer1_attn_dropout = nn.Dropout(p=0.1)
        # LayerNorm: 768
        self.layer1_norm2 = nn.LayerNorm(self.hidden_dim)
        # MLP fc1: 768 → 3072
        self.layer1_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)
        # MLP fc2: 3072 → 768
        self.layer1_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)
        # MLP dropout: p=0.1
        self.layer1_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 2
        # ----------------------------------------------------------------------------
        self.layer2_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer2_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer2_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer2_attn_dropout = nn.Dropout(p=0.1)
        self.layer2_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer2_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer2_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer2_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 3
        # ----------------------------------------------------------------------------
        self.layer3_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer3_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer3_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer3_attn_dropout = nn.Dropout(p=0.1)
        self.layer3_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer3_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer3_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer3_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 4
        # ----------------------------------------------------------------------------
        self.layer4_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer4_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer4_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer4_attn_dropout = nn.Dropout(p=0.1)
        self.layer4_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer4_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer4_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer4_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 5
        # ----------------------------------------------------------------------------
        self.layer5_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer5_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer5_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer5_attn_dropout = nn.Dropout(p=0.1)
        self.layer5_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer5_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer5_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer5_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 6
        # ----------------------------------------------------------------------------
        self.layer6_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer6_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer6_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer6_attn_dropout = nn.Dropout(p=0.1)
        self.layer6_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer6_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer6_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer6_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 7
        # ----------------------------------------------------------------------------
        self.layer7_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer7_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer7_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer7_attn_dropout = nn.Dropout(p=0.1)
        self.layer7_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer7_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer7_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer7_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 8
        # ----------------------------------------------------------------------------
        self.layer8_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer8_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer8_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer8_attn_dropout = nn.Dropout(p=0.1)
        self.layer8_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer8_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer8_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer8_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 9
        # ----------------------------------------------------------------------------
        self.layer9_norm1 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer9_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer9_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer9_attn_dropout = nn.Dropout(p=0.1)
        self.layer9_norm2 = nn.LayerNorm(self.hidden_dim)           # LayerNorm: 768
        self.layer9_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer9_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer9_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 10
        # ----------------------------------------------------------------------------
        self.layer10_norm1 = nn.LayerNorm(self.hidden_dim)          # LayerNorm: 768
        self.layer10_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer10_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer10_attn_dropout = nn.Dropout(p=0.1)
        self.layer10_norm2 = nn.LayerNorm(self.hidden_dim)          # LayerNorm: 768
        self.layer10_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer10_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer10_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 11
        # ----------------------------------------------------------------------------
        self.layer11_norm1 = nn.LayerNorm(self.hidden_dim)          # LayerNorm: 768
        self.layer11_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer11_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer11_attn_dropout = nn.Dropout(p=0.1)
        self.layer11_norm2 = nn.LayerNorm(self.hidden_dim)          # LayerNorm: 768
        self.layer11_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer11_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer11_mlp_dropout = nn.Dropout(p=0.1)
        
        # ----------------------------------------------------------------------------
        # Layer 12
        # ----------------------------------------------------------------------------
        self.layer12_norm1 = nn.LayerNorm(self.hidden_dim)          # LayerNorm: 768
        self.layer12_attn_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)  # QKV: 768 → 2304
        self.layer12_attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)     # Proj: 768 → 768
        self.layer12_attn_dropout = nn.Dropout(p=0.1)
        self.layer12_norm2 = nn.LayerNorm(self.hidden_dim)          # LayerNorm: 768
        self.layer12_mlp_fc1 = nn.Linear(self.hidden_dim, self.mlp_dim)          # fc1: 768 → 3072
        self.layer12_mlp_fc2 = nn.Linear(self.mlp_dim, self.hidden_dim)          # fc2: 3072 → 768
        self.layer12_mlp_dropout = nn.Dropout(p=0.1)
        
        # ============================================================================
        # HEAD: (B, 768) -> (B, num_classes)
        # ============================================================================
        
        # Final LayerNorm: 768
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
        # Classification head: 768 → num_classes
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
            1. QKV projection: (B, N, 768) -> (B, N, 2304) -> split to Q, K, V
            2. Reshape: (B, N, 768) -> (B, 12 heads, N, 64)
            3. Attention: softmax(Q @ K^T / sqrt(64)) @ V
            4. Merge heads: (B, 12, N, 64) -> (B, N, 768)
            5. Output projection: (B, N, 768) -> (B, N, 768)
        
        Args:
            x: Input tensor (B, N, 768)
            qkv_layer: Linear layer for Q, K, V projection
            proj_layer: Linear layer for output projection
            dropout_layer: Dropout for attention weights
            
        Returns:
            Output tensor (B, N, 768)
        """
        B, N, C = x.shape
        
        # QKV projection: (B, N, 768) -> (B, N, 2304) -> (B, N, 3, 12, 64)
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # Permute: (B, N, 3, 12, 64) -> (3, B, 12, N, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, 12, N, 64)
        
        # Scaled dot-product attention
        # attn = softmax(Q @ K^T / sqrt(d_k)) where d_k = 64
        scale = (C // self.num_heads) ** -0.5  # 1/8
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, 12, N, N)
        attn = attn.softmax(dim=-1)
        attn = dropout_layer(attn)
        
        # Apply attention to values: (B, 12, N, N) @ (B, 12, N, 64) -> (B, 12, N, 64)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, 768)
        
        # Output projection: (B, N, 768) -> (B, N, 768)
        x = proj_layer(x)
        
        return x
    
    def _mlp(self, x, fc1, fc2, dropout):
        """
        MLP block with GELU activation
        
        Structure: Linear (768 → 3072) -> GELU -> Dropout -> Linear (3072 → 768) -> Dropout
        
        Args:
            x: Input tensor (B, N, 768)
            fc1: First linear layer (768 → 3072)
            fc2: Second linear layer (3072 → 768)
            dropout: Dropout layer
            
        Returns:
            Output tensor (B, N, 768)
        """
        x = fc1(x)                          # (B, N, 768) -> (B, N, 3072)
        x = torch.nn.functional.gelu(x)     # GELU activation
        x = dropout(x)
        x = fc2(x)                          # (B, N, 3072) -> (B, N, 768)
        x = dropout(x)
        return x
    
    def _transformer_block(self, x, norm1, qkv, proj, attn_drop, norm2, fc1, fc2, mlp_drop):
        """
        Single Transformer encoder block (Pre-LN architecture)
        
        Structure:
            x = x + MHSA(LayerNorm(x))
            x = x + MLP(LayerNorm(x))
            
        Args:
            x: Input tensor (B, N, 768)
            norm1, norm2: LayerNorm layers
            qkv: QKV projection layer
            proj: Output projection layer
            attn_drop: Attention dropout
            fc1, fc2: MLP layers
            mlp_drop: MLP dropout
            
        Returns:
            Output tensor (B, N, 768)
        """
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
        # Conv3d: (B, 3, 32, 256, 256) -> (B, 768, 8, 16, 16)
        # Flatten: (B, 768, 8, 16, 16) -> (B, 768, 2048) -> (B, 2048, 768)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # =========================== CLASS TOKEN ============================
        # Prepend [CLS] token: (B, 2048, 768) -> (B, 2049, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, 768) -> (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # ======================= POSITION EMBEDDING =========================
        # Add learnable position embeddings: (B, 2049, 768) + (1, 2049, 768)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # ====================== TRANSFORMER ENCODER =========================
        # 12 layers: (B, 2049, 768) -> (B, 2049, 768)
        
        # Layer 1
        x = self._transformer_block(x, self.layer1_norm1, self.layer1_attn_qkv, 
                                    self.layer1_attn_proj, self.layer1_attn_dropout,
                                    self.layer1_norm2, self.layer1_mlp_fc1, 
                                    self.layer1_mlp_fc2, self.layer1_mlp_dropout)
        # Layer 2
        x = self._transformer_block(x, self.layer2_norm1, self.layer2_attn_qkv,
                                    self.layer2_attn_proj, self.layer2_attn_dropout,
                                    self.layer2_norm2, self.layer2_mlp_fc1,
                                    self.layer2_mlp_fc2, self.layer2_mlp_dropout)
        # Layer 3
        x = self._transformer_block(x, self.layer3_norm1, self.layer3_attn_qkv,
                                    self.layer3_attn_proj, self.layer3_attn_dropout,
                                    self.layer3_norm2, self.layer3_mlp_fc1,
                                    self.layer3_mlp_fc2, self.layer3_mlp_dropout)
        # Layer 4
        x = self._transformer_block(x, self.layer4_norm1, self.layer4_attn_qkv,
                                    self.layer4_attn_proj, self.layer4_attn_dropout,
                                    self.layer4_norm2, self.layer4_mlp_fc1,
                                    self.layer4_mlp_fc2, self.layer4_mlp_dropout)
        # Layer 5
        x = self._transformer_block(x, self.layer5_norm1, self.layer5_attn_qkv,
                                    self.layer5_attn_proj, self.layer5_attn_dropout,
                                    self.layer5_norm2, self.layer5_mlp_fc1,
                                    self.layer5_mlp_fc2, self.layer5_mlp_dropout)
        # Layer 6
        x = self._transformer_block(x, self.layer6_norm1, self.layer6_attn_qkv,
                                    self.layer6_attn_proj, self.layer6_attn_dropout,
                                    self.layer6_norm2, self.layer6_mlp_fc1,
                                    self.layer6_mlp_fc2, self.layer6_mlp_dropout)
        # Layer 7
        x = self._transformer_block(x, self.layer7_norm1, self.layer7_attn_qkv,
                                    self.layer7_attn_proj, self.layer7_attn_dropout,
                                    self.layer7_norm2, self.layer7_mlp_fc1,
                                    self.layer7_mlp_fc2, self.layer7_mlp_dropout)
        # Layer 8
        x = self._transformer_block(x, self.layer8_norm1, self.layer8_attn_qkv,
                                    self.layer8_attn_proj, self.layer8_attn_dropout,
                                    self.layer8_norm2, self.layer8_mlp_fc1,
                                    self.layer8_mlp_fc2, self.layer8_mlp_dropout)
        # Layer 9
        x = self._transformer_block(x, self.layer9_norm1, self.layer9_attn_qkv,
                                    self.layer9_attn_proj, self.layer9_attn_dropout,
                                    self.layer9_norm2, self.layer9_mlp_fc1,
                                    self.layer9_mlp_fc2, self.layer9_mlp_dropout)
        # Layer 10
        x = self._transformer_block(x, self.layer10_norm1, self.layer10_attn_qkv,
                                    self.layer10_attn_proj, self.layer10_attn_dropout,
                                    self.layer10_norm2, self.layer10_mlp_fc1,
                                    self.layer10_mlp_fc2, self.layer10_mlp_dropout)
        # Layer 11
        x = self._transformer_block(x, self.layer11_norm1, self.layer11_attn_qkv,
                                    self.layer11_attn_proj, self.layer11_attn_dropout,
                                    self.layer11_norm2, self.layer11_mlp_fc1,
                                    self.layer11_mlp_fc2, self.layer11_mlp_dropout)
        # Layer 12
        x = self._transformer_block(x, self.layer12_norm1, self.layer12_attn_qkv,
                                    self.layer12_attn_proj, self.layer12_attn_dropout,
                                    self.layer12_norm2, self.layer12_mlp_fc1,
                                    self.layer12_mlp_fc2, self.layer12_mlp_dropout)
        
        # ============================ HEAD ==================================
        # LayerNorm: (B, 2049, 768) -> (B, 2049, 768)
        x = self.final_norm(x)
        # Extract [CLS] token: (B, 2049, 768) -> (B, 768)
        x = x[:, 0]
        # Classification: (B, 768) -> (B, num_classes)
        x = self.head(x)
        
        return x


def create_model(in_channels: int = 3, num_classes: int = 1) -> ViT3DBase:
    """
    Factory function to create ViT3D-Base model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1)
        
    Returns:
        ViT3DBase model instance
    """
    return ViT3DBase(in_channels=in_channels, num_classes=num_classes)
