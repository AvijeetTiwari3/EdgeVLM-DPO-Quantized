"""
vlm_dpo/training/dpo_loss.py
============================
Closed-Form Multimodal Direct Preference Optimization (DPO) Loss.

Implements the exact DPO objective (Rafailov et al., NeurIPS 2023):
L_DPO = -E [ log sigma( beta * log(pi_theta(yw|x)/pi_ref(yw|x)) - beta * log(pi_theta(yl|x)/pi_ref(yl|x)) ) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class MultimodalDPOLoss(nn.Module):
    """
    Computes DPO Loss and implicit reward metrics for Vision-Language alignment.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def _get_sequence_log_probs(
        self,
        logits: torch.Tensor,       # [Batch, SeqLen, VocabSize]
        labels: torch.Tensor,       # [Batch, SeqLen]
        attention_mask: torch.Tensor # [Batch, SeqLen]
    ) -> torch.Tensor:
        """
        Computes summed log-probabilities over valid token positions.
        """
        # Shift logits and labels for autoregressive language modeling
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log-softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log-probs at target token indices
        gathered_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1) # [Batch, SeqLen - 1]

        # Mask padding and prompt tokens
        masked_log_probs = gathered_log_probs * shift_mask
        summed_log_probs = masked_log_probs.sum(dim=-1) # [Batch]
        return summed_log_probs

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the DPO loss between policy model and frozen reference model.
        """
        # 1. Compute sequence log-probabilities
        pi_logps_w = self._get_sequence_log_probs(policy_chosen_logits, chosen_labels, chosen_mask)
        pi_logps_l = self._get_sequence_log_probs(policy_rejected_logits, rejected_labels, rejected_mask)

        ref_logps_w = self._get_sequence_log_probs(reference_chosen_logits, chosen_labels, chosen_mask)
        ref_logps_l = self._get_sequence_log_probs(reference_rejected_logits, rejected_labels, rejected_mask)

        # 2. Compute log ratio differences (Implicit Rewards)
        # r(x, y) = beta * (log pi(y|x) - log ref(y|x))
        pi_logratios = pi_logps_w - pi_logps_l
        ref_logratios = ref_logps_w - ref_logps_l
        
        logits = self.beta * (pi_logratios - ref_logratios)

        # 3. Bradley-Terry DPO Loss: -log sigma(logits)
        if self.label_smoothing > 0.0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # 4. Compute tracking metrics
        chosen_rewards = self.beta * (pi_logps_w - ref_logps_w).detach()
        rejected_rewards = self.beta * (pi_logps_l - ref_logps_l).detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrics = {
            "dpo_loss": loss.item(),
            "reward_margin": reward_margin,
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
            "preference_accuracy": accuracy
        }

        return loss, metrics
