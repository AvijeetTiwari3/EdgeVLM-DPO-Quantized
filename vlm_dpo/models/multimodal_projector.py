"""
vlm_dpo/models/multimodal_projector.py
======================================
Cross-Modal Vision-to-Language MLP Projector.

Bridges the dimensional and representation gap between continuous visual patch tokens
and discrete language token embeddings.
"""

import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    """
    2-Layer Non-Linear GELU Projector: D_vision -> D_llm.
    """
    def __init__(self, visual_dim: int = 128, llm_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        visual_tokens: [Batch, num_patches, visual_dim]
        Returns: [Batch, num_patches, llm_dim]
        """
        return self.projector(visual_tokens)
