from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset_quality import numeric_punctuation_mismatch_audit_frame, numeric_punctuation_mismatch_count


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")


def test_numeric_punctuation_mismatch_audit_counts_syntax_rule_rows():
    frame = pd.DataFrame(
        [
            {
                "source": "Цена выросла до 12 5 процента.",
                "target": "Цена выросла до 12,5 процента.",
                "rule_ids": '["comma_subordinate"]',
                "source_type": "synthetic_augmented_from_open_clean",
            },
            {
                "source": "Редактор заметил что документ готов.",
                "target": "Редактор заметил, что документ готов.",
                "rule_ids": '["comma_subordinate"]',
                "source_type": "synthetic_augmented_from_open_clean",
            },
        ]
    )

    audit = numeric_punctuation_mismatch_audit_frame(frame)

    assert numeric_punctuation_mismatch_count(frame) == 1
    assert audit["rule_id"].tolist() == ["comma_subordinate"]


def test_canonical_dataset_has_no_numeric_comma_under_syntax_punctuation_rule():
    assert DATASET_PATH.exists(), DATASET_PATH
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    audit = numeric_punctuation_mismatch_audit_frame(df)

    assert audit.empty, audit.head(20).to_dict("records")
