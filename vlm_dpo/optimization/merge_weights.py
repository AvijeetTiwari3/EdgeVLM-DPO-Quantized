"""
vlm_dpo/optimization/merge_weights.py
=====================================
LoRA Adapter Weight Fusion into Base Model Weights.
"""

import torch
import torch.nn as nn
from vlm_dpo.models.qlora_setup import LoRALinear


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Fuses LoRA matrices (B @ A * scaling) directly into the base Linear weight tensor.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.base_layer.weight.data += delta.to(module.base_layer.weight.device)
            module.base_layer.weight.requires_grad = True

    print("[OK] All LoRA adapter weights merged into base layers.")
    return model
