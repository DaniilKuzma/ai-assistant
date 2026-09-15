from __future__ import annotations


def build_direct_edit_heads(
    hidden_size: int,
    token_label_count: int,
    gap_label_count: int,
    boundary_before_label_count: int,
    boundary_after_label_count: int,
    rule_tag_count: int,
):
    import torch.nn as nn

    gap_hidden_size = hidden_size * 3
    return {
        "token_edit": nn.Linear(hidden_size, token_label_count),
        "gap_punctuation": nn.Linear(gap_hidden_size, gap_label_count),
        "boundary_before": nn.Linear(hidden_size, boundary_before_label_count),
        "boundary_after": nn.Linear(hidden_size, boundary_after_label_count),
        "rule": nn.Linear(hidden_size, rule_tag_count),
        "token_confidence": nn.Linear(hidden_size, 1),
        "gap_confidence": nn.Linear(gap_hidden_size, 1),
    }


__all__ = ["build_direct_edit_heads"]

