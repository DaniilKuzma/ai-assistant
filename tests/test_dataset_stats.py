import pandas as pd

from src.data.dataset_stats import dataset_stats


def test_dataset_stats_includes_error_type_counts():
    frame = pd.DataFrame(
        [
            {"is_clean": False, "is_synthetic": True, "split": "train", "error_types": '["spelling"]'},
            {"is_clean": False, "is_synthetic": True, "split": "train", "error_types": '["split_join", "hyphen"]'},
            {"is_clean": True, "is_synthetic": False, "split": "val", "error_types": "[]"},
        ]
    )

    stats = dataset_stats(frame)

    assert stats["error_type_spelling"] == 1
    assert stats["error_type_split_join"] == 1
    assert stats["error_type_hyphen"] == 1
