"""
vlm_dpo/models/qlora_setup.py
=============================
Parameter-Efficient Fine-Tuning (LoRA / QLoRA) for Multimodal Alignment.

Applies low-rank decomposition adapters (r=16, alpha=32) to projection layers.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer: W = W0 + (alpha/r) * (B @ A).
    """
    def __init__(self, base_layer: nn.Linear, r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout)

        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model: nn.Module, r: int = 16, lora_alpha: float = 32.0) -> int:
    """
    Replaces target linear layers in the transformer blocks with LoRALinear modules.
    Returns the number of trainable LoRA parameters.
    """
    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            pass # We can selectively wrap linear layers

    # Wrap projector and causal transformer linear layers
    for i, block in enumerate(model.blocks):
        block.mlp[0] = LoRALinear(block.mlp[0], r=r, lora_alpha=lora_alpha)
        block.mlp[2] = LoRALinear(block.mlp[2], r=r, lora_alpha=lora_alpha)

    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()

    return trainable_params
