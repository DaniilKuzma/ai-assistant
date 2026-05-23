from pathlib import Path

import pandas as pd

from src.data.training_quality_audit import (
    clean_hard_balance_audit_frame,
    clean_hard_balance_counts,
)


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")


def _row(text: str, source_type: str) -> dict[str, object]:
    return {
        "source": text,
        "target": text,
        "split": "train",
        "source_type": source_type,
        "rule_id": "clean_identity" if "clean" in source_type else "clean_identity_hard_negative",
        "rule_ids": '["clean_identity"]',
        "metadata": "{}",
    }


def test_clean_hard_balance_counts_unbalanced_open_clean_rows():
    frame = pd.DataFrame(
        [
            _row('"Open quote only.', "clean_identity_from_open_clean"),
            _row("Text (broken bracket.", "hard_negative_from_open_clean"),
            _row("Normal balanced sentence.", "clean_identity_from_open_clean"),
            {
                **_row('"Broken positive target.', "synthetic_augmented_from_open_clean"),
                "target": '"Broken positive target.',
            },
        ]
    )

    counts = clean_hard_balance_counts(frame)
    audit = clean_hard_balance_audit_frame(frame)

    assert counts["unbalanced_ascii_quotes"] == 1
    assert counts["unbalanced_parentheses"] == 1
    assert counts["unbalanced_guillemets"] == 0
    assert len(audit) == 2
    assert set(audit["source_type"]) == {
        "clean_identity_from_open_clean",
        "hard_negative_from_open_clean",
    }


def test_canonical_clean_and_hard_rows_are_balanced():
    assert DATASET_PATH.exists(), DATASET_PATH
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    counts = clean_hard_balance_counts(df)
    audit = clean_hard_balance_audit_frame(df)

    assert counts == {
        "unbalanced_guillemets": 0,
        "unbalanced_ascii_quotes": 0,
        "unbalanced_parentheses": 0,
        "unbalanced_square_brackets": 0,
        "unbalanced_curly_brackets": 0,
    }
    assert audit.empty, audit.head(20).to_dict("records")
