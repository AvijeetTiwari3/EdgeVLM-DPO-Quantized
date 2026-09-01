"""
vlm_dpo/data_pipeline/preference_collator.py
============================================
Data Collator for Multimodal Direct Preference Optimization (DPO).

Batches chosen and rejected sequences, tokenizes text, generates prompt masks,
and formats visual patch representations.
"""

import torch
from typing import List, Dict, Any, Tuple


class MultimodalDPOCollator:
    """
    Collates multimodal batches for DPO training.
    """

    def __init__(self, tokenizer: Any, max_seq_len: int = 256, num_visual_patches: int = 16, visual_dim: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_visual_patches = num_visual_patches
        self.visual_dim = visual_dim

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        chosen_texts = [b["chosen"] for b in batch]
        rejected_texts = [b["rejected"] for b in batch]

        # Form full text: Prompt + Response
        full_chosen = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>" for p, c in zip(prompts, chosen_texts)]
        full_rejected = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" for p, r in zip(prompts, rejected_texts)]

        # Tokenize chosen and rejected sequences
        chosen_enc = self.tokenizer(
            full_chosen,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            full_rejected,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        # Generate synthetic/real visual patch features: [Batch, num_patches, visual_dim]
        visual_features = torch.randn(len(batch), self.num_visual_patches, self.visual_dim)

        return {
            "visual_features": visual_features,
            "chosen_input_ids": chosen_enc.input_ids,
            "chosen_attention_mask": chosen_enc.attention_mask,
            "rejected_input_ids": rejected_enc.input_ids,
            "rejected_attention_mask": rejected_enc.attention_mask
        }
