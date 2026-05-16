from __future__ import annotations

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
    return stats
