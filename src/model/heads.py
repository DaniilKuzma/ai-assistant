from __future__ import annotations


def build_linear_heads(hidden_size: int, punctuation_labels: int, error_types: int):
    import torch.nn as nn

    return {
        "candidate_projection": nn.Linear(hidden_size * 3, hidden_size),
        "candidate_score": nn.Linear(hidden_size, 1),
        "punctuation_gap": nn.Linear(hidden_size, punctuation_labels),
        "confidence": nn.Linear(hidden_size, 1),
        "error_type": nn.Linear(hidden_size, error_types),
    }
