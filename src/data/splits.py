from __future__ import annotations

import pandas as pd


def add_splits(frame: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 13) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    result = shuffled.copy()
    result["split"] = "test"
    result.loc[: train_end - 1, "split"] = "train"
    result.loc[train_end : val_end - 1, "split"] = "val"
    return result
