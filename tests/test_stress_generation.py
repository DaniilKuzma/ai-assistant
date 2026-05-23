from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.data.corruption_operators import CorruptionResult, Opportunity
from src.data.dataset_contract import DATASET_CONTRACT, LAYER_STRESS_MULTI_ERROR, SYNTHETIC_OPEN_CLEAN
from src.data.operator_dataset_builder import _rule_counts, _stress_count
from src.data.stress_generation import generate_multi_error_stress_rows


@dataclass(frozen=True)
class FakeReplaceOperator:
    rule_id: str
    target_form: str
    error_form: str
    error_type: str = "spelling"
    requires: tuple[str, ...] = ()

    def find_opportunities(self, clean_sentence: str, syntax_analysis=None) -> list[Opportunity]:
        del syntax_analysis
        start = clean_sentence.find(self.target_form)
        if start < 0:
            return []
        return [
            Opportunity(
                start=start,
                end=start + len(self.target_form),
                text=self.target_form,
                rule_id=self.rule_id,
                evidence={"error_form": self.error_form},
                source_type="unit",
            )
        ]

    def corrupt(self, clean_sentence: str, opportunity: Opportunity) -> CorruptionResult:
        source = clean_sentence[: opportunity.start] + self.error_form + clean_sentence[opportunity.end :]
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=[],
            error_bearing_span=(opportunity.start, opportunity.start + len(self.error_form)),
            error_form=self.error_form,
            target_form=opportunity.text,
            generation_strategy="unit_fake",
        )


class FakeRegistry:
    def __init__(self, operators: list[FakeReplaceOperator]) -> None:
        self.operators = {operator.rule_id: operator for operator in operators}

    def get(self, rule_id: str):
        return self.operators.get(rule_id)


class RepairCandidateGenerator:
    def __init__(self, repairs: dict[str, tuple[str, str]]) -> None:
        self.repairs = repairs

    def generate(self, text: str, **_kwargs) -> list[Candidate]:
        candidates: list[Candidate] = []
        for rule_id, (source, replacement) in self.repairs.items():
            start = text.find(source)
            if start < 0:
                continue
            candidates.append(
                Candidate(
                    source=source,
                    replacement=replacement,
                    edit_type="spelling",
                    start=start,
                    end=start + len(source),
                    confidence=1.0,
                    requires_model=False,
                    rule_id=rule_id,
                    mode="candidate_only",
                )
            )
        return candidates


def _metadata(row: dict) -> dict:
    return json.loads(row["metadata"])


def test_stress_row_is_real_multi_error_and_does_not_count_toward_rule_quota():
    target = "Молоко и корова стояли рядом."
    registry = FakeRegistry(
        [
            FakeReplaceOperator("unit_missing_moloko", "Молоко", "Млоко"),
            FakeReplaceOperator("unit_missing_korova", "корова", "корва"),
        ]
    )
    generator = RepairCandidateGenerator(
        {
            "unit_missing_moloko": ("Млоко", "Молоко"),
            "unit_missing_korova": ("корва", "корова"),
        }
    )

    result = generate_multi_error_stress_rows(
        clean_rows=[{"text": target, "source_name": "unit", "source_subcorpus": "stress", "domain": "open_clean"}],
        registry=registry,
        rule_ids=["unit_missing_moloko", "unit_missing_korova"],
        candidate_generator=generator,
        target_count=1,
        seed=7,
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    metadata = _metadata(row)

    assert row["source"] != row["target"]
    assert row["target"] == target
    assert row["gold_edit_count"] >= 2
    assert row["dataset_contract"] == DATASET_CONTRACT
    assert row["dataset_layer"] == LAYER_STRESS_MULTI_ERROR
    assert row["source_type"] == SYNTHETIC_OPEN_CLEAN
    assert row["is_atomic"] is False
    assert row["is_stress"] is True
    assert row["count_toward_rule_quota"] is False
    assert row["loss_weight"] == 0.4
    assert json.loads(row["rule_ids"]) == ["unit_missing_korova", "unit_missing_moloko"]
    assert metadata["stress_rule_ids"] == ["unit_missing_korova", "unit_missing_moloko"]
    assert metadata["generation_strategy"] == "multi_error_stress"
    assert metadata["loss_weight"] == 0.4
    assert metadata["candidate_present"] is True
    assert result.counts_by_rule_combo == {"unit_missing_korova+unit_missing_moloko": 1}


def test_overlapping_opportunities_are_rejected():
    target = "Молоко стояло рядом."
    registry = FakeRegistry(
        [
            FakeReplaceOperator("unit_overlap_a", "Молоко", "Млоко"),
            FakeReplaceOperator("unit_overlap_b", "Молоко", "Молако"),
        ]
    )
    generator = RepairCandidateGenerator(
        {
            "unit_overlap_a": ("Млоко", "Молоко"),
            "unit_overlap_b": ("Молако", "Молоко"),
        }
    )

    result = generate_multi_error_stress_rows(
        clean_rows=[{"text": target}],
        registry=registry,
        rule_ids=["unit_overlap_a", "unit_overlap_b"],
        candidate_generator=generator,
        target_count=1,
        seed=3,
    )

    assert result.rows == []
    assert any(row["reason"] == "overlapping_opportunities" for row in result.rejection_rows)


def test_rule_quota_counter_ignores_explicit_stress_rows():
    stress = {
        "rule_ids": json.dumps(["unit_missing_korova", "unit_missing_moloko"], ensure_ascii=False),
        "dataset_layer": "stress_multi_error",
        "gold_edit_count": 2,
        "count_toward_rule_quota": False,
        "metadata": json.dumps({"count_toward_rule_quota": False}, ensure_ascii=False),
    }
    real_atomic = {
        "rule_ids": json.dumps(["unit_real"], ensure_ascii=False),
        "dataset_layer": "real_atomic",
        "gold_edit_count": 1,
        "count_toward_rule_quota": True,
        "metadata": json.dumps({"count_toward_rule_quota": True}, ensure_ascii=False),
    }
    multi_edit_positive = {
        "rule_ids": json.dumps(["unit_multi"], ensure_ascii=False),
        "dataset_layer": "atomic_positive",
        "gold_edit_count": 2,
        "count_toward_rule_quota": True,
        "metadata": json.dumps({"count_toward_rule_quota": True}, ensure_ascii=False),
    }
    atomic = {
        "rule_ids": json.dumps(["unit_atomic"], ensure_ascii=False),
        "dataset_layer": "atomic_positive",
        "gold_edit_count": 1,
        "count_toward_rule_quota": True,
        "metadata": json.dumps({"count_toward_rule_quota": True}, ensure_ascii=False),
    }

    assert _rule_counts(pd.DataFrame([stress, real_atomic, multi_edit_positive, atomic])) == {"unit_atomic": 1}


def test_stress_counter_requires_stress_layer_and_multi_edit_gold_count():
    metadata_only_fake_stress = {
        "rule_ids": json.dumps(["unit_fake"], ensure_ascii=False),
        "source_type": SYNTHETIC_OPEN_CLEAN,
        "metadata": json.dumps({"is_stress": True}, ensure_ascii=False),
    }
    single_edit_fake_stress = {
        "rule_ids": json.dumps(["unit_single"], ensure_ascii=False),
        "dataset_layer": "stress_multi_error",
        "gold_edit_count": 1,
        "metadata": json.dumps({"is_stress": True, "gold_edit_count": 1}, ensure_ascii=False),
    }
    real_stress = {
        "rule_ids": json.dumps(["unit_a", "unit_b"], ensure_ascii=False),
        "dataset_layer": "stress_multi_error",
        "gold_edit_count": 2,
        "metadata": json.dumps({"is_stress": True, "gold_edit_count": 2}, ensure_ascii=False),
    }

    assert _stress_count(pd.DataFrame([metadata_only_fake_stress, single_edit_fake_stress, real_stress])) == 1
