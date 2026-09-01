"""
vlm_dpo/models/vlm_backbone.py
==============================
Complete On-Device Vision-Language Model (VLM) Architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from vlm_dpo.models.vision_encoder import PatchVisionEncoder
from vlm_dpo.models.multimodal_projector import MultimodalProjector


class CausalTransformerBlock(nn.Module):
    """Transformer Decoder Block with Causal Self-Attention."""
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.ln1(x)
        seq_len = norm_x.size(1)
        if attn_mask is None:
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class EdgeVLM(nn.Module):
    """
    On-Device Vision-Language Model with Prefix Vision Conditioning.
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        visual_dim: int = 64,
        vision_hidden_dim: int = 128,
        llm_hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_visual_patches: int = 16
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.llm_hidden_dim = llm_hidden_dim
        self.num_visual_patches = num_visual_patches

        # 1. Vision Subsystem
        self.vision_encoder = PatchVisionEncoder(
            visual_dim=visual_dim, output_dim=vision_hidden_dim, num_patches=num_visual_patches
        )
        self.projector = MultimodalProjector(
            visual_dim=vision_hidden_dim, llm_dim=llm_hidden_dim
        )

        # 2. Language Subsystem
        self.token_embedding = nn.Embedding(vocab_size, llm_hidden_dim)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(hidden_dim=llm_hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(llm_hidden_dim)
        self.lm_head = nn.Linear(llm_hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        visual_inputs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        visual_inputs: [Batch, num_patches, visual_dim]
        input_ids: [Batch, seq_len]
        Returns: logits of shape [Batch, seq_len, vocab_size]
        """
        vis_features = self.vision_encoder(visual_inputs)
        vis_tokens = self.projector(vis_features)
        text_tokens = self.token_embedding(input_ids)

        combined_seq = torch.cat([vis_tokens, text_tokens], dim=1)

        x = combined_seq
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        text_hidden = x[:, self.num_visual_patches:, :]
        logits = self.lm_head(text_hidden)
        return logits
