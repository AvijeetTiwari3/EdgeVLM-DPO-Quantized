"""
vlm_dpo/models/vision_encoder.py
================================
Vision Patch Encoder for On-Device Vision-Language Models.

Extracts visual patch representations from input images.
"""

import torch
import torch.nn as nn


class PatchVisionEncoder(nn.Module):
    """
    Lightweight Vision Transformer (ViT) patch embedding network.
    """
    def __init__(self, visual_dim: int = 64, output_dim: int = 128, num_patches: int = 16):
        super().__init__()
        self.visual_dim = visual_dim
        self.output_dim = output_dim
        self.num_patches = num_patches

        self.patch_proj = nn.Sequential(
            nn.Linear(visual_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, output_dim) * 0.02)

    def forward(self, visual_inputs: torch.Tensor) -> torch.Tensor:
        """
        visual_inputs: [Batch, num_patches, visual_dim]
        Returns: [Batch, num_patches, output_dim]
        """
        proj = self.patch_proj(visual_inputs)
        return proj + self.pos_embedding
