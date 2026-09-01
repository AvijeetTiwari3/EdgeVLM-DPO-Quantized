"""
tests/test_dpo_loss.py
======================
Unit tests for Multimodal DPO Loss.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_dpo.training.dpo_loss import MultimodalDPOLoss


def test_dpo_loss_computation_and_margin():
    batch_size = 4
    seq_len = 16
    vocab_size = 100
    loss_fn = MultimodalDPOLoss(beta=0.1)

    # 1. Policy model favors chosen tokens over rejected tokens
    policy_chosen = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_rejected = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    # 2. Frozen reference model has neutral logits
    ref_chosen = torch.zeros(batch_size, seq_len, vocab_size)
    ref_rejected = torch.zeros(batch_size, seq_len, vocab_size)

    chosen_labels = torch.randint(1, vocab_size, (batch_size, seq_len))
    chosen_mask = torch.ones(batch_size, seq_len)
    rejected_labels = torch.randint(1, vocab_size, (batch_size, seq_len))
    rejected_mask = torch.ones(batch_size, seq_len)

    loss, metrics = loss_fn(
        policy_chosen_logits=policy_chosen,
        policy_rejected_logits=policy_rejected,
        reference_chosen_logits=ref_chosen,
        reference_rejected_logits=ref_rejected,
        chosen_labels=chosen_labels,
        chosen_mask=chosen_mask,
        rejected_labels=rejected_labels,
        rejected_mask=rejected_mask
    )

    assert loss.item() > 0.0
    assert "reward_margin" in metrics
    assert "preference_accuracy" in metrics

    # Test backward pass
    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None


if __name__ == "__main__":
    test_dpo_loss_computation_and_margin()
    print("[OK] All Multimodal DPO Loss Unit Tests Passed Successfully!")
