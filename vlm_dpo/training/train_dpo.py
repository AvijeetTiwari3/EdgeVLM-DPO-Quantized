"""
vlm_dpo/training/train_dpo.py
=============================
Direct Preference Optimization (DPO) Training Pipeline for Vision-Language Alignment.
"""

import os
import sys
import copy
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vlm_dpo.models.vlm_backbone import EdgeVLM
from vlm_dpo.models.qlora_setup import apply_lora_to_model
from vlm_dpo.training.dpo_loss import MultimodalDPOLoss
from vlm_dpo.data_pipeline.dataset_loader import MultimodalDPODataset
from vlm_dpo.data_pipeline.preference_collator import MultimodalDPOCollator


def train_multimodal_dpo(
    data_path: str = "data_multimodal/multimodal_dpo_pairs.jsonl",
    epochs: int = 1,
    batch_size: int = 4,
    beta: float = 0.1,
    lr: float = 5e-5,
    device: str = "cpu"
):
    print("=" * 70)
    print("   EdgeVLM: MULTIMODAL DIRECT PREFERENCE OPTIMIZATION (DPO) TRAINING")
    print("=" * 70)

    # 1. Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[*] Initializing EdgeVLM Architecture...")
    policy_model = EdgeVLM(
        vocab_size=len(tokenizer),
        visual_dim=64,
        vision_hidden_dim=128,
        llm_hidden_dim=256,
        num_layers=3,
        num_heads=4,
        num_visual_patches=16
    ).to(device)

    # Apply LoRA adapters to policy model
    trainable_params = apply_lora_to_model(policy_model, r=16, lora_alpha=32.0)
    print(f"[OK] LoRA Adapters attached. Trainable parameters: {trainable_params:,}")

    # Create frozen reference model (pi_ref)
    print("[*] Creating frozen reference policy (pi_ref)...")
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    # 2. Ingest Dataset & Collator
    dataset = MultimodalDPODataset(data_path, max_samples=40)
    collator = MultimodalDPOCollator(tokenizer=tokenizer, max_seq_len=128, num_visual_patches=16, visual_dim=64)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    print(f"[OK] Ingested {len(dataset)} real Multimodal DPO preference pairs.")

    # 3. Loss & Optimizer
    dpo_loss_fn = MultimodalDPOLoss(beta=beta)
    optimizer = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    # 4. Training Loop
    policy_model.train()
    print("\n" + "-" * 70)
    print(f"{'Epoch':<8} | {'Step':<8} | {'DPO Loss':<12} | {'Margin':<12} | {'Pref Accuracy':<15}")
    print("-" * 70)

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            vis = batch["visual_features"].to(device)
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)

            policy_chosen_logits = policy_model(vis, c_ids, c_mask)
            policy_rejected_logits = policy_model(vis, r_ids, r_mask)

            with torch.no_grad():
                ref_chosen_logits = reference_model(vis, c_ids, c_mask)
                ref_rejected_logits = reference_model(vis, r_ids, r_mask)

            loss, metrics = dpo_loss_fn(
                policy_chosen_logits=policy_chosen_logits,
                policy_rejected_logits=policy_rejected_logits,
                reference_chosen_logits=ref_chosen_logits,
                reference_rejected_logits=ref_rejected_logits,
                chosen_labels=c_ids,
                chosen_mask=c_mask,
                rejected_labels=r_ids,
                rejected_mask=r_mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 2 == 0 or (step + 1) == len(train_loader):
                print(f"#{epoch+1:<7} | #{step+1:<7} | {metrics['dpo_loss']:<12.4f} | {metrics['reward_margin']:<12.4f} | {metrics['preference_accuracy']*100:<14.1f}%")

    print("-" * 70)
    print("\n[OK] Multimodal DPO Alignment Completed Successfully!")
    return policy_model, tokenizer
