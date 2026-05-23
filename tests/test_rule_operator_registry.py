from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.corruption_operators import (
    CorruptionOperator,
    CorruptionResult,
    HardNegativeResult,
    Opportunity,
    RuleOperatorRegistry,
    VerificationResult,
    resolve_operator_training_targets,
    write_operator_registry_reports,
)


@dataclass(frozen=True)
class DummyOperator(CorruptionOperator):
    rule_id: str = "dummy_rule"
    error_type: str = "spelling"
    requires: tuple[str, ...] = ()

    def find_opportunities(self, clean_sentence: str, syntax_analysis=None) -> list[Opportunity]:
        index = clean_sentence.find("молоко")
        if index < 0:
            return []
        return [
            Opportunity(
                start=index,
                end=index + len("молоко"),
                text="молоко",
                rule_id=self.rule_id,
                evidence={"target_form": "молоко"},
                syntax_family="dictionary",
                confidence=1.0,
                source_type="unit",
            )
        ]

    def corrupt(self, clean_sentence: str, opportunity: Opportunity) -> CorruptionResult:
        source = clean_sentence[: opportunity.start] + "млоко" + clean_sentence[opportunity.end :]
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=[],
            error_bearing_span=(opportunity.start, opportunity.start + len("млоко")),
            error_form="млоко",
            target_form=opportunity.text,
            generation_strategy="corpus_opportunity",
            metadata={"operator": "dummy"},
        )

    def verify(self, source: str, target: str, opportunity: Opportunity) -> VerificationResult:
        passed = source != target and "млоко" in source and "молоко" in target
        return VerificationResult(
            passed=passed,
            reason="ok" if passed else "not_dummy_corruption",
            actual_error_family="dictionary",
            expected_error_family="dictionary",
            candidate_present=passed,
            candidate_rule_ids=[self.rule_id] if passed else [],
            target_quality_pass=True,
            semantic_alignment_pass=passed,
        )

    def generate_hard_negatives(self, clean_sentence: str, syntax_analysis=None) -> list[HardNegativeResult]:
        return [
            HardNegativeResult(
                source=clean_sentence,
                target=clean_sentence,
                rule_id=self.rule_id,
                reason="clean_identity",
                metadata={"operator": "dummy"},
            )
        ]


def test_rule_operator_registry_registers_and_retrieves_operator():
    registry = RuleOperatorRegistry()
    operator = DummyOperator()

    registry.register(operator)

    assert registry.get("dummy_rule") is operator
    assert registry.has_operator("dummy_rule")
    assert registry.rule_ids() == ["dummy_rule"]


def test_rule_without_operator_is_excluded_and_reported(tmp_path: Path):
    registry = RuleOperatorRegistry()
    registry.register(DummyOperator())

    rows = resolve_operator_training_targets(
        candidate_rule_ids=["dummy_rule", "missing_rule"],
        registry=registry,
        excluded_rule_ids={},
    )
    write_operator_registry_reports(rows, registry=registry, reports_dir=tmp_path)

    by_rule = {row["rule_id"]: row for row in rows}
    assert by_rule["dummy_rule"]["include"] is True
    assert by_rule["missing_rule"]["include"] is False
    assert by_rule["missing_rule"]["reason"] == "BLOCK_NO_OPERATOR"

    without = pd.read_csv(tmp_path / "rules_without_operator.csv")
    assert without["rule_id"].tolist() == ["missing_rule"]


def test_dummy_operator_makes_new_rule_dataset_eligible_without_builder_changes():
    registry = RuleOperatorRegistry()
    registry.register(DummyOperator(rule_id="new_dictionary_rule"))

    rows = resolve_operator_training_targets(
        candidate_rule_ids=["new_dictionary_rule"],
        registry=registry,
        excluded_rule_ids={},
    )

    assert rows == [
        {
            "rule_id": "new_dictionary_rule",
            "include": True,
            "reason": "registered_operator",
            "operator": "DummyOperator",
            "error_type": "spelling",
            "requires": "",
        }
    ]

