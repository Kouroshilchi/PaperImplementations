import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nlp.Transformers.Transformers import *

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embedding_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embedding_dim, 
                             kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)                    # (B, embedding_dim, H/P, W/P)
        x = x.flatten(2)                    # (B, embedding_dim, num_patches)
        x = x.transpose(1, 2)               # (B, num_patches, embedding_dim)
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.pos_embed
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoderViT(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, int(embedding_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm.transpose(0, 1), x_norm.transpose(0, 1), x_norm.transpose(0, 1))
        x = x + attn_output.transpose(0, 1)
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x
    
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 embedding_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embedding_dim)
        self.encoder = nn.Sequential(*[
            TransformerEncoderViT(embedding_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x) 
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token) 
        return logits
    
if __name__ == "__main__":
    model = ViT(img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                embedding_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1).to('cuda')
    summary(model, (3, 224, 224))