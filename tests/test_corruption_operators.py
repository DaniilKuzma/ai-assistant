from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.corruption_operators import build_default_operator_registry


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")


def test_syntax_style_operator_uses_same_registry_interface():
    registry = build_default_operator_registry()
    operator = registry.get("comma_subordinate")
    assert operator is not None

    target = "Редактор заметил, что документ готов утром."
    opportunities = operator.find_opportunities(target)

    assert opportunities
    result = operator.corrupt(target, opportunities[0])
    verification = operator.verify(result.source, result.target, opportunities[0])

    assert result.source == "Редактор заметил что документ готов утром."
    assert verification.passed
    assert verification.semantic_alignment_pass


def test_dictionary_style_operator_uses_same_registry_interface():
    registry = build_default_operator_registry()
    operator = registry.get("missing_letter_candidate")
    assert operator is not None

    target = "В отчете комиссии встретилось слово молоко сегодня."
    opportunities = operator.find_opportunities(target)

    assert opportunities
    result = operator.corrupt(target, opportunities[0])
    verification = operator.verify(result.source, result.target, opportunities[0])

    assert result.source != result.target
    assert "млоко" in result.source
    assert verification.passed
    assert verification.expected_error_family == "missing_letter"


def test_operator_verification_rejects_identity_pair():
    registry = build_default_operator_registry()
    operator = registry.get("comma_subordinate")
    opportunity = operator.find_opportunities("Редактор заметил, что документ готов утром.")[0]

    verification = operator.verify(
        "Редактор заметил, что документ готов утром.",
        "Редактор заметил, что документ готов утром.",
        opportunity,
    )

    assert not verification.passed
    assert verification.reason == "identity_pair"


def test_canonical_synthetic_rows_record_operator_verification_pass():
    assert DATASET_PATH.exists(), DATASET_PATH
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    synthetic = df[df["source_type"].astype(str).eq("synthetic_augmented_from_open_clean")]

    assert not synthetic.empty
    for raw in synthetic["metadata"].head(5000):
        metadata = json.loads(raw)
        assert metadata["operator_verify_passed"] is True
        assert metadata["candidate_present"] is True
        assert metadata["semantic_alignment_pass"] is True
        assert metadata["target_quality_pass"] is True
        assert metadata["operator_verification"]
        assert all(item["passed"] is True for item in metadata["operator_verification"].values())
