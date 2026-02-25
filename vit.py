import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=3,          
            out_channels=256,
            kernel_size=16,
            stride=16
        )

    def forward(self, x):
        x = self.patch_embed(x)     # [B, 256, 14, 14]
        x = x.flatten(2)            # [B, 256, 196]
        x = x.transpose(1, 2)       # [B, 196, 256]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(256)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        
        x = x + self.multi_head_attention(
            self.layer_norm1(x),
            self.layer_norm1(x),
            self.layer_norm1(x)
        )[0]

        
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbedding()

        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.positional_embedding = nn.Parameter(
            torch.randn(1, 197, 256)  
        )

        
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoder() for _ in range(6)]
        )

        self.norm = nn.LayerNorm(256)

        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2)         
        )

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.positional_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        cls_output = x[:, 0]
        return self.classifier(cls_output)
