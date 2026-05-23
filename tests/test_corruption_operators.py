from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.data.corruption_operators import (
    CorruptionResult,
    Opportunity,
    RuleOperatorRegistry,
    _verification,
    build_default_operator_registry,
)
from src.data.operator_dataset_builder import generate_atomic_positive_rows_from_clean_pool, _verified_synthetic_rows


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")


class EmptyCandidateGenerator:
    def generate(self, text: str):
        del text
        return []


class MissingCandidateOperator:
    rule_id = "comma_subordinate"
    error_type = "punctuation"
    requires = ("syntax",)

    def verify(self, source: str, target: str, opportunity: Opportunity):
        return _verification(self.rule_id, source, target, opportunity, candidate_generator=EmptyCandidateGenerator())


class UnitCandidateGenerator:
    def generate(self, text: str):
        start = text.find("млоко")
        if start >= 0:
            return [
                Candidate(
                    source="млоко",
                    replacement="молоко",
                    edit_type="spelling",
                    start=start,
                    end=start + len("млоко"),
                    rule_id="unit_atomic",
                )
            ]
        return []


class MultiEditCleanPoolOperator:
    rule_id = "unit_atomic"
    error_type = "spelling"
    requires = ()

    def find_opportunities(self, clean_sentence: str, syntax_analysis=None):
        del syntax_analysis
        if "молоко" not in clean_sentence:
            return []
        return [Opportunity(clean_sentence.index("молоко"), clean_sentence.index("молоко") + len("молоко"), "молоко", self.rule_id)]

    def corrupt(self, clean_sentence: str, opportunity: Opportunity):
        first_start = opportunity.start
        first_end = opportunity.end
        source = clean_sentence[:first_start] + "млоко" + clean_sentence[first_end:]
        second_start = source.index("комиссии")
        second_end = second_start + len("комиссии")
        source = source[:second_start] + "кмиссии" + source[second_end:]
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=[
                {
                    "source": "млоко",
                    "replacement": "молоко",
                    "edit_type": "spelling_replace",
                    "start": first_start,
                    "end": first_start + len("млоко"),
                },
                {
                    "source": "кмиссии",
                    "replacement": "комиссии",
                    "edit_type": "spelling_replace",
                    "start": second_start,
                    "end": second_start + len("кмиссии"),
                },
            ],
            error_bearing_span=(first_start, first_start + len("млоко")),
            error_form="млоко",
            target_form="молоко",
            generation_strategy="unit_multi_edit",
        )

    def verify(self, source: str, target: str, opportunity: Opportunity):
        return _verification(self.rule_id, source, target, opportunity, candidate_generator=UnitCandidateGenerator())


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
    assert verification.gold_edit_count == 1
    assert verification.strict_validator_passed
    assert verification.matched_candidate is not None


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
    assert verification.gold_edit_count == 1
    assert verification.strict_validator_passed
    assert verification.matched_candidate is not None


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


def test_operator_verification_does_not_trust_candidate_present_evidence():
    source = "Редактор заметил что документ готов утром."
    target = "Редактор заметил, что документ готов утром."
    opportunity = Opportunity(
        start=-1,
        end=-1,
        text="",
        rule_id="comma_subordinate",
        evidence={"candidate_present": True, "candidate_rule_ids": ["comma_subordinate"]},
        syntax_family="comma_subordinate",
        source_type="unit",
    )

    verification = _verification(
        "comma_subordinate",
        source,
        target,
        opportunity,
        candidate_generator=EmptyCandidateGenerator(),
    )

    assert not verification.passed
    assert verification.reason == "candidate_missing"
    assert verification.candidate_present is False


def test_operator_verification_rejects_multi_edit_pair_as_non_atomic():
    source = "Сегодня я незнаю что делать после проверки отчета"
    target = "Сегодня я не знаю, что делать после проверки отчета."
    opportunity = Opportunity(0, 0, "", "ne_verb", source_type="unit")

    verification = _verification("ne_verb", source, target, opportunity)

    assert not verification.passed
    assert verification.reason == "non_atomic_edit_count"
    assert verification.gold_edit_count > 1


def test_verified_synthetic_rows_reports_candidate_missing_despite_fake_metadata():
    registry = RuleOperatorRegistry()
    registry.register(MissingCandidateOperator())
    frame = pd.DataFrame(
        [
            {
                "source": "Редактор заметил что документ готов утром.",
                "target": "Редактор заметил, что документ готов утром.",
                "source_type": "synthetic_augmented_from_open_clean",
                "rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
                "metadata": json.dumps(
                    {"candidate_present": True, "target_family": "comma_subordinate"},
                    ensure_ascii=False,
                ),
            }
        ]
    )

    rows, rejections = _verified_synthetic_rows(frame, {"comma_subordinate"}, registry)

    assert rows == []
    assert rejections
    assert rejections[0]["reason"] == "candidate_missing"


def test_atomic_positive_generation_rejects_multi_edit_operator_result(tmp_path: Path):
    clean_pool_path = tmp_path / "clean_sentence_pool.csv.gz"
    pd.DataFrame(
        [
            {
                "text": "В отчете комиссии встретилось слово молоко сегодня.",
                "source_name": "unit",
                "source_subcorpus": "unit",
                "domain": "unit",
                "sentence_id": "s1",
                "hash": "h1",
            }
        ]
    ).to_csv(clean_pool_path, index=False)
    registry = RuleOperatorRegistry()
    registry.register(MultiEditCleanPoolOperator())

    result = generate_atomic_positive_rows_from_clean_pool(
        clean_pool_path,
        registry,
        {"unit_atomic"},
        {
            "data": {
                "clean_pool_chunksize": 1,
                "rule_quota": {
                    "min_atomic_positives_per_active_rule": 1,
                    "preferred_atomic_positives_per_active_rule": 1,
                    "max_total_per_rule_id": 1,
                },
            }
        },
        UnitCandidateGenerator(),
    )

    assert result.rows == []
    assert any(row["rule_id"] == "unit_atomic" and row["reason"] == "non_atomic_edit_count" for row in result.rejection_rows)


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
