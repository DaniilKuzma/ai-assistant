from __future__ import annotations

import json
from typing import Any

import pandas as pd


def dataset_stats(frame: pd.DataFrame) -> dict[str, int]:
    stats = {
        "total": int(len(frame)),
        "clean": int(frame.get("is_clean", pd.Series(dtype=bool)).fillna(False).sum()),
        "synthetic": int(frame.get("is_synthetic", pd.Series(dtype=bool)).fillna(False).sum()),
    }
    stats["real"] = stats["total"] - stats["clean"] - stats["synthetic"]
    if "split" in frame.columns:
        for split, count in frame["split"].value_counts().to_dict().items():
            stats[f"split_{split}"] = int(count)
    if "error_types" in frame.columns:
        for error_type, count in _error_type_counts(frame["error_types"]).items():
            stats[f"error_type_{error_type}"] = count
    return stats


def _error_type_counts(values: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        for error_type in _parse_error_types(value):
            counts[error_type] = counts.get(error_type, 0) + 1
    return dict(sorted(counts.items()))


def _parse_error_types(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
        return [str(parsed)] if str(parsed) else []
    return [str(value)] if str(value) else []
