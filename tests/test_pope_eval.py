"""
tests/test_pope_eval.py
=======================
Unit tests for POPE Hallucination Evaluator.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_dpo.evaluation.pope_evaluator import POPEEvaluator
from vlm_dpo.evaluation.nlp_metrics import compute_rouge_l, compute_bleu_4


def test_pope_metric_computation():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_multimodal", "pope_adversarial.json")
    evaluator = POPEEvaluator(data_path)
    assert len(evaluator.samples) > 0


def test_nlp_metrics_computation():
    ref = "The medical scan reveals clear pulmonary opacity."
    hyp = "The medical scan reveals pulmonary opacity in lungs."

    rouge = compute_rouge_l(ref, hyp)
    bleu = compute_bleu_4(ref, hyp)

    assert rouge > 0.5
    assert bleu > 0.3


if __name__ == "__main__":
    test_pope_metric_computation()
    test_nlp_metrics_computation()
    print("[OK] All Evaluation & Benchmark Unit Tests Passed Successfully!")
