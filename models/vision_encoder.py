"""
Vision Encoder using Vision Transformer
"""
import torch
import torch.nn as nn
import timm


class VisualEncoder(nn.Module):
    """Vision Transformer-based visual feature encoder."""

    def __init__(self, model_name: str = "vit_base_patch16_224", out_dim: int = 512):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.vit.reset_classifier(0)

        self.proj = nn.Sequential(
            nn.Linear(self.vit.num_features, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features."""
        feat = self.vit(x)
        out = self.proj(feat)
        return out
