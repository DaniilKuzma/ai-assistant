from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def threshold_sweep(scores: Iterable[dict], thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    items = list(scores)
    total_gold_edits = max((int(item.get("total_gold_edits", 0)) for item in items), default=0)
    if total_gold_edits <= 0:
        total_gold_edits = sum(1 for item in items if item.get("is_correct", False))
    for threshold in thresholds:
        selected = [item for item in items if item.get("confidence", 0.0) >= threshold]
        true_positive = sum(item.get("is_correct", False) for item in selected)
        precision = 0.0 if not selected else true_positive / len(selected)
        recall = 0.0 if total_gold_edits == 0 else true_positive / total_gold_edits
        rows.append({"threshold": threshold, "precision": precision, "recall": recall})
    return pd.DataFrame(rows)
