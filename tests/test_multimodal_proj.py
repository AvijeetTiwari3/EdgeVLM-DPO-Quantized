"""
tests/test_multimodal_proj.py
=============================
Unit tests for Vision Encoder, Projector & EdgeVLM Backbone.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_dpo.models.vision_encoder import PatchVisionEncoder
from vlm_dpo.models.multimodal_projector import MultimodalProjector
from vlm_dpo.models.vlm_backbone import EdgeVLM


def test_vision_encoder_and_projector_dimensions():
    batch_size = 2
    num_patches = 16
    vis_dim = 64
    vis_hidden = 128
    llm_dim = 256

    encoder = PatchVisionEncoder(visual_dim=vis_dim, output_dim=vis_hidden, num_patches=num_patches)
    projector = MultimodalProjector(visual_dim=vis_hidden, llm_dim=llm_dim)

    raw_vis = torch.randn(batch_size, num_patches, vis_dim)
    enc_out = encoder(raw_vis)
    assert enc_out.shape == (batch_size, num_patches, vis_hidden)

    proj_out = projector(enc_out)
    assert proj_out.shape == (batch_size, num_patches, llm_dim)


def test_edge_vlm_forward_pass():
    batch_size = 2
    seq_len = 12
    vocab_size = 1000
    model = EdgeVLM(vocab_size=vocab_size, visual_dim=64, vision_hidden_dim=128, llm_hidden_dim=256, num_layers=2)

    vis_in = torch.randn(batch_size, 16, 64)
    text_in = torch.randint(1, vocab_size, (batch_size, seq_len))

    logits = model(vis_in, text_in)
    # Output must match text sequence length
    assert logits.shape == (batch_size, seq_len, vocab_size)


if __name__ == "__main__":
    test_vision_encoder_and_projector_dimensions()
    test_edge_vlm_forward_pass()
    print("[OK] All Vision-Language Architecture Unit Tests Passed Successfully!")
