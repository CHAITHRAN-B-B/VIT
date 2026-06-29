"""
vit.py — Model factory functions for every architecture variant used in this project.

Architecture summary
--------------------
Family A  – Custom VisionTransformer (vision_clean / vision_combined)
  Checkpoints : best_vit_model.pth, best_vit_combined.pth, best_vit_combined1.pth,
                best_vit_model1.pth
  Backbone    : hand-rolled ViT with NoiseExtractor / FourierMagnitudeExtractor /
                LightingConvergenceExtractor fused into patch embeddings
  Head        : Linear(emd_dim, mlp_nodes) → ReLU → Linear(mlp_nodes, 2)
  NOTE: These weights are NOT compatible with torchvision's vit_b_16.

Family B  – torchvision vit_b_16, plain Linear head (create_vitv5/6/7)
  Checkpoints : best_vit_v5.pth, best_vit_v5-1.pth, best_vit_v5-2.pth,
                best_vit_v6.pth (not present in /models but same arch),
                best_vit_v7.pth, best_vit_v7-1.pth
  Head        : Linear(768, 2)

Family C  – torchvision vit_b_16, Dropout(0.5)+Linear head (vitv8/v10)
  Checkpoints : best_vit_v8.pth, best_vit_v10.pth
  Head        : Dropout(0.5) → Linear(768, 2)

Usage
-----
  from vit import build_model_for_checkpoint, build_vit_v10
  model = build_model_for_checkpoint("VIT/models/best_vit_v8.pth")
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Shared constants for the custom (Family-A) architecture ───────────────────
_PATCH_SIZE   = 16
_IMAGE_SIZE   = 224
_NUM_PATCHES  = (_IMAGE_SIZE // _PATCH_SIZE) ** 2   # 196
_EMD_DIM      = 256
_MLP_HIDDEN   = 512   # actual MLP hidden dim in all saved checkpoints
_CLF_HIDDEN   = 512   # actual classifier hidden dim in all saved checkpoints
_NUM_HEADS    = 8
_BLOCKS       = 8     # actual block count in all saved checkpoints
_NUM_CLASSES  = 2


# ─────────────────────────────────────────────────────────────────────────────
# Family A — Custom hand-rolled VisionTransformer
#
# Checkpoint key names were reverse-engineered directly from the saved .pth
# files — attribute names MUST match exactly for load_state_dict() to work.
# ─────────────────────────────────────────────────────────────────────────────

class _PatchEmbedding(nn.Module):
    """
    Checkpoint key: patch_embedding.patch_embed.*
    No LayerNorm — the checkpoint has no patch_embedding.norm.* keys.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # attribute MUST be named patch_embed (not proj)
        self.patch_embed = nn.Conv2d(
            in_channels, _EMD_DIM,
            kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE,
        )

    def forward(self, x):
        x = self.patch_embed(x)               # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)   # [B, N, D]


class _TransformerEncoder(nn.Module):
    """
    Checkpoint key pattern:
      transformer_blocks.X.layer_norm1.*
      transformer_blocks.X.layer_norm2.*
      transformer_blocks.X.multi_head_attention.*
      transformer_blocks.X.mlp.*
    MLP: Linear(256, 512) → GELU → Linear(512, 256)
    """
    def __init__(self):
        super().__init__()
        # MUST be named layer_norm1 / layer_norm2
        self.layer_norm1 = nn.LayerNorm(_EMD_DIM)
        self.layer_norm2 = nn.LayerNorm(_EMD_DIM)
        # MUST be named multi_head_attention
        self.multi_head_attention = nn.MultiheadAttention(
            _EMD_DIM, _NUM_HEADS, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(_EMD_DIM, _MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(_MLP_HIDDEN, _EMD_DIM),
        )

    def forward(self, x):
        y, _ = self.multi_head_attention(
            self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x)
        )
        x = x + y
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _CustomVisionTransformer(nn.Module):
    """
    Matches: best_vit_model.pth / best_vit_combined.pth / best_vit_combined1.pth
    These checkpoints have NO extractor sub-modules; patch_embed in_channels = 3.

    Key layout:
      cls_token                     (1, 1, 256)
      positional_embedding          (1, 197, 256)
      patch_embedding.patch_embed.* (256, 3, 16, 16)
      transformer_blocks.0-7.*
      norm.*
      classifier.0/2.*
    """
    def __init__(self):
        super().__init__()
        self.cls_token            = nn.Parameter(torch.randn(1, 1, _EMD_DIM) * 0.02)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, _NUM_PATCHES + 1, _EMD_DIM) * 0.02
        )
        self.patch_embedding    = _PatchEmbedding(in_channels=3)
        self.transformer_blocks = nn.ModuleList(
            [_TransformerEncoder() for _ in range(_BLOCKS)]
        )
        self.norm       = nn.LayerNorm(_EMD_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(_EMD_DIM, _CLF_HIDDEN),
            nn.ReLU(),
            nn.Linear(_CLF_HIDDEN, _NUM_CLASSES),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)                      # [B, 196, 256]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat((cls, x), dim=1)                 # [B, 197, 256]
        x   = x + self.positional_embedding
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])


class _CustomVisionTransformerV2(nn.Module):
    """
    Matches: best_vit_model1.pth only.
    This checkpoint adds registered buffers for noise / fourier / lighting
    extractors AND uses 6-channel patch_embed (3 RGB + 3 extractor channels).

    Key layout:
      noise_extractor.weight             (1,1,3,3)
      fourier_extractor.window_2d        (1,1,224,224)
      fourier_extractor.center_mask      (1,1,224,224)
      fourier_extractor.weight           (1,1,3,3)
      light_extractor.sobel_x/y          (1,1,3,3)
      light_extractor.cos/sin_anchors    (1,8,1,1)
      patch_embedding.patch_embed.*      (256, 6, 16, 16)
      cls_token / positional_embedding
      transformer_blocks.0-7.*
      norm.*  /  classifier.*
    """
    def __init__(self):
        super().__init__()
        # ── Extractor buffers (registered so they appear in state_dict) ────────
        self.noise_extractor   = _NoiseExtractorV2()
        self.fourier_extractor = _FourierExtractorV2()
        self.light_extractor   = _LightExtractorV2()

        self.cls_token            = nn.Parameter(torch.randn(1, 1, _EMD_DIM) * 0.02)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, _NUM_PATCHES + 1, _EMD_DIM) * 0.02
        )
        self.patch_embedding    = _PatchEmbedding(in_channels=6)  # 3+1+1+1
        self.transformer_blocks = nn.ModuleList(
            [_TransformerEncoder() for _ in range(_BLOCKS)]
        )
        self.norm       = nn.LayerNorm(_EMD_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(_EMD_DIM, _CLF_HIDDEN),
            nn.ReLU(),
            nn.Linear(_CLF_HIDDEN, _NUM_CLASSES),
        )

    def forward(self, x):
        B = x.size(0)
        noise   = self.noise_extractor(x)
        fourier = self.fourier_extractor(x)
        light   = self.light_extractor(x)
        x_combined = torch.cat([x, noise, fourier, light], dim=1)  # [B,6,H,W]
        x = self.patch_embedding(x_combined)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat((cls, x), dim=1)
        x   = x + self.positional_embedding
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])


# ── Sub-modules for _CustomVisionTransformerV2 (buffer-based extractors) ──────

class _NoiseExtractorV2(nn.Module):
    """Checkpoint key: noise_extractor.weight  shape (1,1,3,3)."""
    def __init__(self):
        super().__init__()
        # Learned 3×3 conv stored as a buffer named 'weight'
        self.register_buffer('weight', torch.ones(1, 1, 3, 3) / 9.0)

    def forward(self, x):
        import torch.nn.functional as F
        gray = x.mean(dim=1, keepdim=True)
        blur = F.conv2d(gray, self.weight, padding=1)
        return gray - blur


class _FourierExtractorV2(nn.Module):
    """Checkpoint keys: fourier_extractor.window_2d / center_mask / weight."""
    def __init__(self):
        super().__init__()
        H = W = _IMAGE_SIZE
        self.register_buffer('window_2d',   torch.ones(1, 1, H, W))
        self.register_buffer('center_mask', torch.ones(1, 1, H, W))
        self.register_buffer('weight',      torch.ones(1, 1, 3, 3) / 9.0)

    def forward(self, x):
        gray    = x.mean(dim=1, keepdim=True)
        fft     = torch.fft.fft2(gray * self.window_2d)
        mag     = torch.abs(fft) + 1e-8
        log_mag = torch.log(mag)
        m_mean  = log_mag.mean(dim=(-2, -1), keepdim=True)
        m_std   = log_mag.std(dim=(-2, -1), keepdim=True)
        thr     = m_mean + 1.5 * m_std
        return torch.where(log_mag < thr, thr, log_mag)


class _LightExtractorV2(nn.Module):
    """Checkpoint keys: light_extractor.sobel_x/y / cos_anchors / sin_anchors."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        angles  = torch.linspace(0, 3.14159, 8).view(1, 8, 1, 1)
        self.register_buffer('sobel_x',     sobel_x)
        self.register_buffer('sobel_y',     sobel_y)
        self.register_buffer('cos_anchors', torch.cos(angles))
        self.register_buffer('sin_anchors', torch.sin(angles))

    def forward(self, x):
        import torch.nn.functional as F
        gray = x.mean(dim=1, keepdim=True)
        gx   = F.conv2d(gray, self.sobel_x, padding=1)
        gy   = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Family B — torchvision ViT-B/16 with plain Linear head  (v5, v6, v7)
# ─────────────────────────────────────────────────────────────────────────────

def build_vit_linear_head(num_classes: int = 2) -> nn.Module:
    """
    torchvision vit_b_16 with head = Linear(768, num_classes).

    Matches: best_vit_v5.pth, best_vit_v5-1.pth, best_vit_v5-2.pth,
             best_vit_v7.pth, best_vit_v7-1.pth
    (create_vitv5.py / create_vitv6.py / create_vitv7.py)
    """
    model = models.vit_b_16()
    in_features = model.heads.head.in_features  # 768
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Family C — torchvision ViT-B/16 with Dropout(0.5)+Linear head  (v8, v10)
# ─────────────────────────────────────────────────────────────────────────────

def build_vit_v10(num_classes: int = 2) -> nn.Module:
    """
    torchvision vit_b_16 with head = Dropout(0.5) → Linear(768, num_classes).

    Matches: best_vit_v8.pth, best_vit_v10.pth
    (vitv8.ipynb / vitv10.ipynb)

    Weights are NOT loaded here — caller is responsible for load_state_dict().
    """
    model = models.vit_b_16()
    in_features = model.heads.head.in_features  # 768
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


# Alias kept for any existing imports
build_vit_v8 = build_vit_v10


# ─────────────────────────────────────────────────────────────────────────────
# Unified auto-selector — pick the right builder from the checkpoint filename
# ─────────────────────────────────────────────────────────────────────────────

def build_model_for_checkpoint(checkpoint_path: str,
                                num_classes: int = 2) -> nn.Module:
    """
    Return an un-loaded model whose architecture matches *checkpoint_path*.

    Rules (matched on the basename):
      • Contains 'v8' or 'v10'               → Family C (Dropout+Linear head)
      • Contains 'v5', 'v6', or 'v7'         → Family B (plain Linear head)
      • Contains 'combined' or 'model'        → Family A (custom VisionTransformer)
      • Falls back to Family C (most recent)

    Caller must still call:
        model.load_state_dict(torch.load(checkpoint_path, ...))
        model.to(device).eval()
    """
    import os
    name = os.path.basename(checkpoint_path).lower()

    if "v10" in name or "v8" in name:
        return build_vit_v10(num_classes)

    if "v5" in name or "v6" in name or "v7" in name:
        return build_vit_linear_head(num_classes)

    if "model1" in name:
        # best_vit_model1.pth — 6-channel patch embed + extractor buffers
        return _CustomVisionTransformerV2()

    if "combined" in name or "model" in name:
        # best_vit_combined.pth / best_vit_combined1.pth / best_vit_model.pth
        # — plain 3-channel, no extractor sub-modules
        return _CustomVisionTransformer()

    # Default fallback
    return build_vit_v10(num_classes)


# ---------------------------------------------------------------------------
# Backward-compat shim: keeps old `VisionTransformer` name importable.
# ---------------------------------------------------------------------------
VisionTransformer = build_vit_v10
