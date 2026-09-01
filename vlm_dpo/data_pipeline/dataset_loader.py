"""
vlm_dpo/data_pipeline/dataset_loader.py
=======================================
Multimodal Dataset Parser for Real-World DPO Preference Pairs & Chart Benchmarks.
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset


class MultimodalDPODataset(Dataset):
    """
    Parses paired preference samples: (Image Features / IDs, Prompt, Chosen Response, Rejected Response).
    """

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = 1000):
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        self.samples.append(item)
                        if max_samples and len(self.samples) >= max_samples:
                            break
                    except Exception:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.samples[idx]


class ChartQABenchmarkDataset(Dataset):
    """
    Parses real ChartQA human reasoning questions.
    """

    def __init__(self, json_path: str, max_samples: Optional[int] = 500):
        self.samples = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.samples = data[:max_samples] if max_samples else data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.samples[idx]
