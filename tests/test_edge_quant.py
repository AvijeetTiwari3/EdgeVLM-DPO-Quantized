"""
tests/test_edge_quant.py
========================
Unit tests for Edge Quantization & Memory Reduction.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_dpo.models.vlm_backbone import EdgeVLM
from vlm_dpo.optimization.edge_quantizer import EdgeQuantizer


def test_tensor_quantization_and_dequantization():
    w = torch.randn(64, 64)

    # INT8 Quantization
    q8, scale8 = EdgeQuantizer.quantize_tensor_int8(w)
    deq8 = EdgeQuantizer.dequantize(q8, scale8)
    assert q8.dtype == torch.int8
    assert torch.allclose(w, deq8, atol=0.1)

    # INT4 Quantization
    q4, scale4 = EdgeQuantizer.quantize_tensor_int4(w)
    deq4 = EdgeQuantizer.dequantize(q4, scale4)
    assert q4.dtype == torch.int8
    assert (q4 >= -8).all() and (q4 <= 7).all()


def test_model_compression_stats():
    model = EdgeVLM(vocab_size=500, visual_dim=32, vision_hidden_dim=64, llm_hidden_dim=128, num_layers=2)
    stats = EdgeQuantizer.compress_model(model, bits=4)

    assert stats["compression_ratio"] > 1.5
    assert stats["memory_reduction_pct"] > 40.0


if __name__ == "__main__":
    test_tensor_quantization_and_dequantization()
    test_model_compression_stats()
    print("[OK] All Edge Quantization Unit Tests Passed Successfully!")
