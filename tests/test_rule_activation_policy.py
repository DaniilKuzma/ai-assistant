from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset_contract import ensure_contract_columns
from src.data.operator_dataset_builder import _rule_counts as operator_rule_counts
from src.rules.capabilities import (
    RuleActivationPolicy,
    RuleCapability,
    activation_policy_from_config,
    activation_stage_for_rule,
    active_rule_ids_for_training,
    capability_manifest_fields,
    expanded_activation_blocked_rows,
    expanded_activation_candidate_rows,
    production_ready_rule_ids,
    training_candidate_rule_ids,
)


def _capability(
    rule_id: str,
    decision: str,
    *,
    candidate: bool = True,
    synthetic: bool = True,
    hard_negative: bool = True,
    validator: bool = True,
    requires: list[str] | None = None,
    ner: bool = True,
) -> RuleCapability:
    return RuleCapability(
        taxonomy_key=f"fixture_{rule_id}",
        domain="test",
        entry_type="leaf_rule",
        title=rule_id,
        orfogrammka_id="",
        project_rule_ids=[rule_id],
        implementation_status="model_required",
        requires=requires or [],
        executable=True,
        training_eligible=decision == "INCLUDE_NOW",
        training_decision=decision,
        training_reason=decision,
        has_candidate_path=candidate,
        has_synthetic_support=synthetic,
        has_hard_negative_support=hard_negative,
        has_validator_support=validator,
        has_dictionary_support=True,
        has_syntax_support=True,
        has_morphology_support=True,
        has_ner_support=ner,
        risk_level="low",
    )


def _expanded_policy(**overrides: object) -> RuleActivationPolicy:
    config = {
        "data": {
            "rule_activation": {
                "mode": "expanded_safe",
                "include_decisions": [
                    "INCLUDE_NOW",
                    "INCLUDE_AFTER_THRESHOLD_CALIBRATION",
                    "INCLUDE_AFTER_TRAINING",
                ],
                "include_after_validator_when_runtime_supported": True,
                "require_candidate_path": True,
                "require_synthetic_support": True,
                "require_hard_negative_support": True,
                "require_validator_support_or_empirical_pass": True,
                "exclude_blocked": True,
                "exclude_ner_required_without_ner": True,
                "min_empirical_validator_passes": 3,
                "max_empirical_validator_examples": 10,
                "expected_min_production_ready_rule_count": 1,
                "expected_min_training_candidate_rule_count": 1,
                "target_training_candidate_rule_count": 4,
                "fail_below_min_training_candidate_rule_count": True,
                "warn_below_target_training_candidate_rule_count": True,
            }
        }
    }
    config["data"]["rule_activation"].update(overrides)
    return activation_policy_from_config(config)


def test_strict_mode_returns_only_include_now_rules():
    capabilities = [
        _capability("prod_rule", "INCLUDE_NOW"),
        _capability("threshold_rule", "INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
        _capability("training_rule", "INCLUDE_AFTER_TRAINING"),
    ]

    assert active_rule_ids_for_training(capabilities) == ["prod_rule"]
    assert training_candidate_rule_ids(capabilities) == ["prod_rule"]


def test_expanded_safe_includes_threshold_and_training_decisions_with_gates():
    capabilities = [
        _capability("prod_rule", "INCLUDE_NOW"),
        _capability("threshold_rule", "INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
        _capability("training_rule", "INCLUDE_AFTER_TRAINING"),
        _capability("no_candidate", "INCLUDE_AFTER_TRAINING", candidate=False),
    ]
    policy = _expanded_policy()

    assert active_rule_ids_for_training(capabilities, policy=policy) == [
        "prod_rule",
        "threshold_rule",
        "training_rule",
    ]
    assert activation_stage_for_rule(capabilities[1], policy) == "threshold_calibration"
    assert activation_stage_for_rule(capabilities[2], policy) == "needs_training"


def test_include_after_validator_requires_runtime_validator_or_empirical_probe(monkeypatch):
    capabilities = [
        _capability("runtime_validator", "INCLUDE_AFTER_VALIDATOR", validator=True),
        _capability("probe_pass", "INCLUDE_AFTER_VALIDATOR", validator=False),
        _capability("probe_fail", "INCLUDE_AFTER_VALIDATOR", validator=False),
    ]
    policy = _expanded_policy(include_decisions=["INCLUDE_NOW"])

    def fake_probe(capability: RuleCapability, current_policy: RuleActivationPolicy) -> dict[str, object]:
        del current_policy
        passed = capability.project_rule_ids == ["probe_pass"]
        return {
            "rule_id": capability.project_rule_ids[0],
            "status": "pass" if passed else "fail",
            "passed_examples": 3 if passed else 0,
            "example_count": 3,
            "reason": "" if passed else "validator_probe_failed",
            "empirical_validator_support": passed,
        }

    monkeypatch.setattr("src.rules.capabilities._empirical_validator_probe_for_capability", fake_probe)

    assert active_rule_ids_for_training(capabilities, policy=policy) == ["probe_pass", "runtime_validator"]
    fields = capability_manifest_fields(capabilities, policy=policy)
    assert fields["validator_probe_summary"]["passed_rule_ids"] == ["probe_pass"]
    assert fields["validator_probe_summary"]["failed_rule_reasons"]["probe_fail"] == "validator_probe_failed"


def test_blocked_and_ner_required_rules_never_become_active():
    capabilities = [
        _capability("blocked", "BLOCK_PLANNED"),
        _capability("ner_rule", "INCLUDE_AFTER_TRAINING", requires=["ner"], ner=False),
        _capability("safe_rule", "INCLUDE_AFTER_TRAINING"),
    ]
    policy = _expanded_policy()

    assert active_rule_ids_for_training(capabilities, policy=policy) == ["safe_rule"]


def test_production_ready_rules_are_subset_of_training_candidates():
    capabilities = [
        _capability("prod_rule", "INCLUDE_NOW"),
        _capability("threshold_rule", "INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
    ]
    policy = _expanded_policy()

    production = set(production_ready_rule_ids(capabilities))
    candidates = set(training_candidate_rule_ids(capabilities, policy=policy))

    assert production == {"prod_rule"}
    assert production <= candidates


def test_policy_manifest_flags_training_candidate_count_below_minimum():
    capabilities = [_capability("prod_rule", "INCLUDE_NOW")]
    policy = _expanded_policy(expected_min_training_candidate_rule_count=2)

    fields = capability_manifest_fields(capabilities, policy=policy)

    assert fields["production_ready_rule_count"] == 1
    assert fields["training_candidate_rule_count"] == 1
    assert "training_candidate_rule_count_below_min:1<2" in fields["activation_policy"]["errors"]


def test_policy_manifest_flags_final_active_count_thresholds():
    capabilities = [
        _capability("prod_rule", "INCLUDE_NOW"),
        _capability("threshold_rule", "INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
    ]
    policy = _expanded_policy(
        expected_min_final_active_rule_count=3,
        target_final_active_rule_count=4,
        fail_below_final_active_rule_count=True,
        warn_below_target_final_active_rule_count=True,
    )

    fields = capability_manifest_fields(capabilities, policy=policy, final_active_rule_ids=["prod_rule"])

    assert fields["final_active_rule_count"] == 1
    assert fields["target_final_active_rule_count"] == 4
    assert "final_active_rule_count_below_min:1<3" in fields["activation_policy"]["errors"]
    assert "final_active_rule_count_below_target:1<4" in fields["activation_policy"]["warnings"]


def test_expanded_activation_candidate_rows_include_report_gates_and_validator_probe(monkeypatch):
    capabilities = [
        _capability("prod_rule", "INCLUDE_NOW"),
        _capability("threshold_rule", "INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
        _capability("validator_rule", "INCLUDE_AFTER_VALIDATOR", validator=False),
        _capability("blocked_rule", "BLOCK_PLANNED"),
    ]
    policy = _expanded_policy(include_decisions=["INCLUDE_NOW"])

    def fake_probe(capability: RuleCapability, current_policy: RuleActivationPolicy) -> dict[str, object]:
        del current_policy
        return {
            "rule_id": capability.project_rule_ids[0],
            "status": "pass",
            "passed_examples": 3,
            "example_count": 3,
            "reason": "",
            "empirical_validator_support": True,
        }

    monkeypatch.setattr("src.rules.capabilities._empirical_validator_probe_for_capability", fake_probe)

    rows = expanded_activation_candidate_rows(
        capabilities,
        policy=policy,
        evidence_by_rule={
            "prod_rule": {"generated_probe_count": 4, "verifier_pass_count": 4, "candidate_recall": 1.0},
            "threshold_rule": {"generated_probe_count": 3, "verifier_pass_count": 3, "candidate_recall": 1.0},
            "validator_rule": {"generated_probe_count": 3, "verifier_pass_count": 3, "candidate_recall": 1.0},
        },
        final_active_rule_ids=["prod_rule", "threshold_rule", "validator_rule"],
    )
    by_rule = {row["rule_id"]: row for row in rows}

    assert by_rule["prod_rule"]["activation_bucket"] == "production_ready"
    assert by_rule["threshold_rule"]["activation_bucket"] == "threshold_calibration"
    assert by_rule["validator_rule"]["activation_bucket"] == "validator_dependent"
    assert by_rule["validator_rule"]["empirical_validator_support"] is True
    assert by_rule["validator_rule"]["included"] is True
    assert by_rule["blocked_rule"]["included"] is False
    assert by_rule["prod_rule"]["generated_probe_count"] == 4
    assert by_rule["prod_rule"]["verifier_pass_count"] == 4
    assert by_rule["prod_rule"]["candidate_recall"] == 1.0


def test_expanded_activation_blocked_rows_explain_gate_failures():
    capabilities = [
        _capability("ner_rule", "INCLUDE_AFTER_TRAINING", requires=["ner"], ner=False),
        _capability("quote_open", "INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
        _capability("mining_rule", "MINING_ONLY"),
    ]
    policy = _expanded_policy()

    blocked = expanded_activation_blocked_rows(capabilities, policy=policy)
    by_rule = {row["rule_id"]: row for row in blocked}

    assert by_rule["ner_rule"]["blocker"] == "needs_NER"
    assert by_rule["quote_open"]["blocker"] == "broad_normalization_bucket"
    assert by_rule["mining_rule"]["blocker"] == "MINING_ONLY"


def test_dataset_contract_adds_activation_columns():
    frame = pd.DataFrame(
        [
            {
                "source": "Автор незнает ответ.",
                "target": "Автор не знает ответ.",
                "source_type": "synthetic_augmented_from_open_clean",
                "dataset_layer": "atomic_positive",
                "rule_id": "ne_verb",
                "rule_ids": json.dumps(["ne_verb"], ensure_ascii=False),
                "edits": json.dumps([{"source": "незнает", "replacement": "не знает"}], ensure_ascii=False),
                "activation_stage": "threshold_calibration",
                "production_ready": False,
            }
        ]
    )

    upgraded = ensure_contract_columns(frame)

    assert upgraded.loc[0, "activation_stage"] == "threshold_calibration"
    assert bool(upgraded.loc[0, "production_ready"]) is False
    assert operator_rule_counts(upgraded) == {"ne_verb": 1}
