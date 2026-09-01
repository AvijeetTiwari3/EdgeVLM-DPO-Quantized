"""
vlm_dpo/optimization/edge_quantizer.py
======================================
Hardware-Aware INT4 / INT8 Symmetric Weight-Only Quantization (W4A16).

Compresses Vision-Language Model weights by >70% for real-time edge execution
(Apple Silicon, CPU, mobile NPU).
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class EdgeQuantizer:
    """
    Symmetric Linear Quantization Engine for Edge VLM Deployment.
    """

    @staticmethod
    def quantize_tensor_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes FP32/FP16 weight tensor to signed INT8 (-128 to 127).
        Formula: Q = round(W / Scale), Scale = max(|W|) / 127
        """
        max_val = torch.max(torch.abs(weight))
        scale = max_val / 127.0 if max_val > 0 else torch.tensor(1.0)
        q_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
        return q_weight, scale

    @staticmethod
    def quantize_tensor_int4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes FP32/FP16 weight tensor to signed INT4 (-8 to 7).
        Formula: Q = round(W / Scale), Scale = max(|W|) / 7
        """
        max_val = torch.max(torch.abs(weight))
        scale = max_val / 7.0 if max_val > 0 else torch.tensor(1.0)
        q_weight = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)
        return q_weight, scale

    @classmethod
    def dequantize(cls, q_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantizes INT4/INT8 back to float for forward computation."""
        return q_weight.float() * scale

    @classmethod
    def compress_model(cls, model: nn.Module, bits: int = 4) -> Dict[str, Any]:
        """
        Compresses all linear weights in the model and calculates memory reduction.
        """
        original_size_bytes = 0
        quantized_size_bytes = 0
        quantized_state = {}

        for name, param in model.named_parameters():
            num_elements = param.numel()
            orig_bytes = num_elements * param.element_size()
            original_size_bytes += orig_bytes

            if "weight" in name and param.dim() >= 2:
                if bits == 4:
                    q_w, scale = cls.quantize_tensor_int4(param.data)
                    q_bytes = num_elements * 0.5 # 4-bits = 0.5 byte
                else:
                    q_w, scale = cls.quantize_tensor_int8(param.data)
                    q_bytes = num_elements * 1.0 # 8-bits = 1.0 byte

                quantized_state[name] = {"q_weight": q_w, "scale": scale}
                quantized_size_bytes += int(q_bytes)
            else:
                quantized_state[name] = param.data
                quantized_size_bytes += orig_bytes

        orig_mb = original_size_bytes / (1024 * 1024)
        quant_mb = quantized_size_bytes / (1024 * 1024)
        reduction_pct = (1.0 - (quant_mb / orig_mb)) * 100.0 if orig_mb > 0 else 0.0

        return {
            "quantized_state": quantized_state,
            "original_size_mb": round(orig_mb, 2),
            "quantized_size_mb": round(quant_mb, 2),
            "compression_ratio": round(orig_mb / quant_mb, 2) if quant_mb > 0 else 1.0,
            "memory_reduction_pct": round(reduction_pct, 2)
        }
