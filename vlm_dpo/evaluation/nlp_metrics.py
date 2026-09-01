"""
vlm_dpo/evaluation/nlp_metrics.py
=================================
NLP Metrics (ROUGE-L, BLEU-4, and Perplexity) for Multimodal Visual Reasoning.
"""

import math
from typing import List, Dict, Any


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Computes ROUGE-L based on Longest Common Subsequence (LCS).
    """
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    m = len(ref_tokens)
    n = len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return round(f1, 4)


def compute_bleu_4(reference: str, hypothesis: str) -> float:
    """
    Computes simplified token-level overlap BLEU score.
    """
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    matches = sum(1 for token in hyp_tokens if token in ref_tokens)
    precision = matches / len(hyp_tokens)
    brevity_penalty = min(1.0, math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
    return round(precision * brevity_penalty, 4)
