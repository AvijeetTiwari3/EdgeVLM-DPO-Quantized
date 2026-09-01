"""
vlm_dpo/evaluation/edge_profiler.py
===================================
Hardware & Latency Profiler for On-Device VLM Inference.

Measures Time-To-First-Token (TTFT), Inter-Token Latency (ITL / Tokens per sec),
and Memory Footprint.
"""

import time
import torch
from typing import Dict, Any, List, Optional


class EdgeHardwareProfiler:
    """
    On-Device hardware efficiency and latency benchmark harness.
    """

    @staticmethod
    def profile_inference(
        model: Any,
        tokenizer: Any,
        prompt: str = "Explain the trend in this financial chart:",
        num_patches: int = 16,
        visual_dim: int = 64,
        max_gen_tokens: int = 32,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Measures TTFT, Inter-Token Latency, and Throughput.
        """
        model.eval()
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc.input_ids
        vis_inputs = torch.randn(1, num_patches, visual_dim, device=device)

        # Warm-up pass
        with torch.no_grad():
            _ = model(vis_inputs, input_ids)

        # Timed Generation Loop
        start_time = time.perf_counter()
        first_token_time = None
        current_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(max_gen_tokens):
                step_start = time.perf_counter()
                logits = model(vis_inputs, current_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_tok], dim=-1)

                if first_token_time is None:
                    first_token_time = time.perf_counter()

        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        itl_ms = ((total_time - (ttft_ms / 1000.0)) / max(1, max_gen_tokens - 1)) * 1000.0
        tok_per_sec = max_gen_tokens / total_time if total_time > 0 else 0.0

        return {
            "prompt": prompt,
            "generated_tokens": max_gen_tokens,
            "total_time_ms": round(total_time * 1000.0, 2),
            "time_to_first_token_ms": round(ttft_ms, 2),
            "inter_token_latency_ms": round(itl_ms, 2),
            "tokens_per_second": round(tok_per_sec, 2),
            "device": device
        }
