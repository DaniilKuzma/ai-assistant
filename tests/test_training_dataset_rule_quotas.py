import json
import os
from pathlib import Path

import yaml

from src.candidates.candidate_generator import Candidate
from src.data.training_dataset import compute_broad_dataset_targets, resolve_broad_active_training_rules
from src.data.operator_dataset_builder import (
    LAYER_ATOMIC_POSITIVE,
    _final_active_rule_ids_after_gates,
    _chunk_rule_ids_for_workers,
    _dataset_build_workers,
    _merge_parallel_atomic_positive_results,
    _operator_rule_quota_config,
    _pre_gate_atomic_counts_by_rule,
    _pre_gate_hard_negative_counts_by_rule,
    _prune_failed_diversity_rules,
    _requested_layer_targets_from_config,
    _resolve_effective_layer_targets,
    _rule_counts as _operator_rule_counts,
    generate_atomic_positive_rows_from_syntax_synthetic,
)
from src.data._training_dataset_builder import (
    CLEAN_IDENTITY_OPEN,
    HARD_NEGATIVE_OPEN,
    REAL_ERROR_PAIR,
    SYNTHETIC_OPEN_CLEAN,
    _finalize_quota_state,
    _quality_source_minimum_targets,
    _remaining_fallback_template_budget,
)
from src.rules.syntax_synthetic import SUPPORTED_SYNTAX_RULE_IDS
from src.rules.capabilities import (
    active_rule_ids_for_training,
    activation_policy_from_config,
    activation_policy_rows,
    load_rule_capabilities,
)

import pandas as pd


def _config() -> dict:
    return yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))


class _CommaCandidateGenerator:
    def generate(self, text: str):
        candidates = []
        for marker in ("готов", "завершен"):
            index = text.find(marker)
            if index < 0:
                continue
            start = index + len(marker)
            candidates.append(
                Candidate(
                    source="",
                    replacement=",",
                    edit_type="punctuation_insert",
                    start=start,
                    end=start,
                    rule_id="comma_subordinate",
                    syntax_family="subordinate_clause_comma",
                )
            )
        return candidates


def _patch_syntax_eval_rows(monkeypatch, rows: list[dict[str, object]]) -> None:
    import src.rules.syntax_synthetic as syntax_module

    monkeypatch.setattr(syntax_module, "SUPPORTED_SYNTAX_RULE_IDS", ("comma_subordinate",))
    monkeypatch.setattr(
        syntax_module,
        "build_syntax_eval_examples",
        lambda **_kwargs: pd.DataFrame(rows),
    )


def _syntax_config() -> dict:
    return {
        "data": {
            "rule_quota": {
                "min_atomic_positives_per_active_rule": 1,
                "preferred_atomic_positives_per_active_rule": 2,
                "max_total_per_rule_id": 2,
            }
        },
        "nlp": {"syntax": {"enabled": True}},
    }


def test_syntax_synthetic_atomic_positive_rows_are_verified(monkeypatch, tmp_path: Path):
    _patch_syntax_eval_rows(
        monkeypatch,
        [
            {
                "source": "Когда отчет готов мы отправим письмо утром.",
                "target": "Когда отчет готов, мы отправим письмо утром.",
                "rule_id": "comma_subordinate",
                "syntax_family": "subordinate_clause_comma",
                "source_type": "syntax_synthetic_eval",
                "candidate_present": False,
                "candidate_rule_ids": "[]",
                "hard_negative": False,
                "metadata": json.dumps({"candidate_present": False}, ensure_ascii=False),
            }
        ],
    )

    rows = generate_atomic_positive_rows_from_syntax_synthetic(
        ["comma_subordinate", "unit_atomic"],
        tmp_path / "missing_clean_pool.csv.gz",
        _CommaCandidateGenerator(),
        _syntax_config(),
    )

    assert len(rows) == 1
    row = rows[0]
    metadata = json.loads(row["metadata"])
    assert row["dataset_contract"] == "candidate_opportunity"
    assert row["dataset_layer"] == "atomic_positive"
    assert row["source_type"] == "synthetic_augmented_from_open_clean"
    assert row["activation_source"] == "syntax_synthetic"
    assert row["rule_id"] == "comma_subordinate"
    assert json.loads(row["rule_ids"]) == ["comma_subordinate"]
    assert row["source"] != row["target"]
    assert int(row["gold_edit_count"]) == 1
    assert bool(row["count_toward_rule_quota"]) is True
    assert row["verification_status"] == "passed"
    assert metadata["candidate_present"] is True
    assert metadata["strict_validator_passed"] is True
    assert metadata["target_quality_pass"] is True
    assert metadata["extra_edit_count"] == 0
    assert metadata["generation_sources"] == ["syntax_synthetic"]


def test_syntax_synthetic_does_not_trust_fake_candidate_metadata(monkeypatch, tmp_path: Path):
    _patch_syntax_eval_rows(
        monkeypatch,
        [
            {
                "source": "Когда отчет готов мы отправим письмо утром.",
                "target": "Когда отчет готов, мы отправим письмо утром.",
                "rule_id": "comma_subordinate",
                "syntax_family": "subordinate_clause_comma",
                "source_type": "syntax_synthetic_eval",
                "candidate_present": True,
                "candidate_rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
                "hard_negative": False,
                "metadata": json.dumps(
                    {"candidate_present": True, "target_family": "comma_subordinate"},
                    ensure_ascii=False,
                ),
            }
        ],
    )

    rows = generate_atomic_positive_rows_from_syntax_synthetic(
        ["comma_subordinate"],
        tmp_path / "missing_clean_pool.csv.gz",
        candidate_generator=type("NoCandidateGenerator", (), {"generate": lambda self, _text: []})(),
        config=_syntax_config(),
    )

    assert rows == []


def test_syntax_synthetic_rejects_multi_edit_targets(monkeypatch, tmp_path: Path):
    _patch_syntax_eval_rows(
        monkeypatch,
        [
            {
                "source": "Когда отчет готов мы отправим письмо и когда архив завершен мы обновим журнал.",
                "target": "Когда отчет готов, мы отправим письмо и когда архив завершен, мы обновим журнал.",
                "rule_id": "comma_subordinate",
                "syntax_family": "subordinate_clause_comma",
                "source_type": "syntax_synthetic_eval",
                "candidate_present": True,
                "candidate_rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
                "hard_negative": False,
                "metadata": json.dumps(
                    {"candidate_present": True, "target_family": "comma_subordinate"},
                    ensure_ascii=False,
                ),
            }
        ],
    )

    rows = generate_atomic_positive_rows_from_syntax_synthetic(
        ["comma_subordinate"],
        tmp_path / "missing_clean_pool.csv.gz",
        _CommaCandidateGenerator(),
        _syntax_config(),
    )

    assert rows == []


def test_broad_active_training_rules_include_syntax_and_legacy_candidate_backed_rules():
    rows = resolve_broad_active_training_rules(_config())
    active = {row["rule_id"]: row for row in rows if row["include"]}
    config = _config()
    capability_active = set(active_rule_ids_for_training(load_rule_capabilities("configs/rules.yaml"), policy=activation_policy_from_config(config)))

    assert set(active) == capability_active
    assert set(SUPPORTED_SYNTAX_RULE_IDS) <= set(active)
    assert "quote_pair_balance" in active
    assert {
        "hyphen_whitelist",
        "comma_subordinate",
        "subject_predicate_dash",
        "homogeneous_comma",
        "direct_speech_dash",
        "address_comma",
    } <= set(active)

    excluded = {row["rule_id"]: row["reason"] for row in rows if not row["include"]}
    assert {"dictionary_fuzzy", "final_punctuation_default", "ne_verb"} <= set(active)
    assert "capitalization_ner" in excluded
    assert "needs_NER" in excluded["capitalization_ner"]
    assert "quote_open" in excluded
    assert "broad_normalization" in excluded["quote_open"]
    assert "quote_close" in excluded
    assert "broad_normalization" in excluded["quote_close"]
    assert "yo_e_candidate" in excluded
    assert "disabled" in excluded["yo_e_candidate"]


def test_broad_active_training_rule_quotas_follow_dataset_plan():
    rows = resolve_broad_active_training_rules(_config())
    active = {row["rule_id"]: row for row in rows if row["include"]}

    assert active["comma_subordinate"]["target_min_examples"] == 1500
    assert active["comma_subordinate"]["target_preferred_examples"] == 2500
    assert active["subject_predicate_dash"]["target_min_examples"] == 1500
    assert active["subject_predicate_dash"]["target_preferred_examples"] == 2500
    assert active["hyphen_whitelist"]["target_min_examples"] == 1000
    assert active["hyphen_whitelist"]["target_preferred_examples"] == 3000


def test_syntax_disabled_config_blocks_syntax_required_training_candidates():
    config = _config()
    config["nlp"]["syntax"]["enabled"] = False
    capabilities = load_rule_capabilities("configs/rules.yaml", config=config)
    policy = activation_policy_from_config(config)
    active = set(active_rule_ids_for_training(capabilities, policy=policy))
    activation = {row["rule_id"]: row for row in activation_policy_rows(capabilities, policy=policy)}

    assert "comma_subordinate" not in active
    assert activation["comma_subordinate"]["activation_exclusion_reason"] == "BLOCK_NEEDS_SYNTAX"


def test_dynamic_targets_scale_from_active_rule_quotas_and_real_pair_count():
    rows = resolve_broad_active_training_rules(_config())
    targets = compute_broad_dataset_targets(rows, real_pair_count=1236)
    config = _config()
    capability_active_count = len(active_rule_ids_for_training(load_rule_capabilities("configs/rules.yaml"), policy=activation_policy_from_config(config)))

    assert targets["active_rule_count"] == capability_active_count
    assert targets["targeted_synthetic_base"] == sum(
        row["target_preferred_examples"] for row in rows if row["include"]
    )
    assert targets["targeted_synthetic_target"] >= targets["targeted_synthetic_base"]
    assert targets["total_target"] >= 200000
    assert targets["split_sizes"]["train"] == int(targets["total_target"] * 0.8)
    assert targets["split_sizes"]["val"] == int(targets["total_target"] * 0.1)
    assert targets["split_sizes"]["test"] == targets["total_target"] - targets["split_sizes"]["train"] - targets["split_sizes"]["val"]
    assert targets["clean_identity_target"] >= int(targets["total_target"] * 0.10)
    assert targets["hard_negative_target"] >= int(targets["total_target"] * 0.10)
    assert 0.03 <= targets["multi_error_stress_target"] / targets["total_target"] <= 0.05


def test_effective_atomic_target_adjusts_to_final_active_rule_capacity():
    config = {
        "data": {
            "composition": {
                "atomic_positive_ratio": 0.45,
                "atomic_hard_negative_ratio": 0.35,
                "clean_identity_ratio": 0.12,
                "stress_multi_error_ratio": 0.05,
                "real_atomic_train_ratio": 0.03,
                "allow_layer_target_adjustment": True,
                "fail_on_unadjusted_layer_deficit": True,
            },
            "rule_quota": {
                "min_atomic_positives_per_active_rule": 1000,
                "preferred_atomic_positives_per_active_rule": 2500,
            },
        }
    }
    requested = _requested_layer_targets_from_config(config, 200000)
    quota = _operator_rule_quota_config(config)

    effective, adjustments, warnings, audit_errors = _resolve_effective_layer_targets(
        config,
        requested,
        final_active_rule_count=43,
        quota_config=quota,
    )

    assert requested[LAYER_ATOMIC_POSITIVE] == 90000
    assert effective[LAYER_ATOMIC_POSITIVE] == 90000
    assert adjustments == {}
    assert warnings == []
    assert audit_errors == []


def test_effective_atomic_target_records_adjustment_when_requested_exceeds_capacity():
    config = {
        "data": {
            "composition": {
                "atomic_positive_ratio": 0.45,
                "atomic_hard_negative_ratio": 0.35,
                "clean_identity_ratio": 0.12,
                "stress_multi_error_ratio": 0.05,
                "real_atomic_train_ratio": 0.03,
                "allow_layer_target_adjustment": True,
                "fail_on_unadjusted_layer_deficit": True,
            },
            "rule_quota": {
                "min_atomic_positives_per_active_rule": 1000,
                "preferred_atomic_positives_per_active_rule": 1500,
            },
        }
    }
    requested = _requested_layer_targets_from_config(config, 200000)
    quota = _operator_rule_quota_config(config)

    effective, adjustments, warnings, audit_errors = _resolve_effective_layer_targets(
        config,
        requested,
        final_active_rule_count=43,
        quota_config=quota,
    )

    assert requested[LAYER_ATOMIC_POSITIVE] == 90000
    assert effective[LAYER_ATOMIC_POSITIVE] == 64500
    assert adjustments[LAYER_ATOMIC_POSITIVE] == {
        "requested": 90000,
        "feasible": 64500,
        "selected": 64500,
        "reason": "requested_exceeds_per_rule_capacity",
    }
    assert warnings == ["atomic_positive_target_adjusted"]
    assert audit_errors == []


def test_effective_atomic_target_can_use_explicit_raised_max_without_adjustment():
    config = {
        "data": {
            "composition": {
                "atomic_positive_ratio": 0.45,
                "atomic_hard_negative_ratio": 0.35,
                "clean_identity_ratio": 0.12,
                "stress_multi_error_ratio": 0.05,
                "real_atomic_train_ratio": 0.03,
                "allow_layer_target_adjustment": True,
            },
            "rule_quota": {
                "min_atomic_positives_per_active_rule": 1000,
                "preferred_atomic_positives_per_active_rule": 1500,
                "max_total_per_rule_id": 3000,
            },
        }
    }
    requested = _requested_layer_targets_from_config(config, 200000)
    quota = _operator_rule_quota_config(config)

    effective, adjustments, warnings, audit_errors = _resolve_effective_layer_targets(
        config,
        requested,
        final_active_rule_count=43,
        quota_config=quota,
    )

    assert effective[LAYER_ATOMIC_POSITIVE] == 90000
    assert adjustments == {}
    assert warnings == []
    assert audit_errors == []


def test_excluded_active_rules_do_not_remain_underfilled_after_quota_finalize():
    frame = pd.DataFrame(
        [
            {
                "rule_ids": '["safe_rule"]',
                "split": "train",
                "error_type": "spelling",
                "dataset_layer": "atomic_positive",
                "count_toward_rule_quota": True,
                "gold_edit_count": 1,
                "edits": json.dumps([{"rule_id": "safe_rule"}]),
            },
            {
                "rule_ids": '["safe_rule"]',
                "split": "val",
                "error_type": "spelling",
                "dataset_layer": "atomic_positive",
                "count_toward_rule_quota": True,
                "gold_edit_count": 1,
                "edits": json.dumps([{"rule_id": "safe_rule"}]),
            },
        ]
    )
    quota_state = {
        "active_rule_ids": ["safe_rule", "excluded_rule"],
        "excluded_active_rule_ids": ["excluded_rule"],
        "quota_rows": [
            {"rule_id": "safe_rule", "target_min_total": 2, "preferred_total": 2},
            {"rule_id": "excluded_rule", "target_min_total": 1000, "preferred_total": 2500},
        ],
        "excluded_rows": [
            {
                "rule_id": "excluded_rule",
                "reason": "excluded_after_quality_gate",
                "new_status": "excluded_from_synthetic_target",
            }
        ],
    }

    result = _finalize_quota_state(
        frame,
        quota_state,
        quota_config={"min_total_per_active_rule": 2, "preferred_total_per_active_rule": 2},
        cap_config={"generation_rule_cap": 10, "generation_error_type_cap": 10, "rule_max_totals": {}},
    )

    assert result["active_rule_ids"] == ["safe_rule"]
    assert result["low_count_active_rule_ids"] == []


def test_candidate_opportunity_active_quota_counts_only_atomic_positive_layer():
    frame = pd.DataFrame(
        [
            {
                "rule_ids": json.dumps(["unit_atomic"], ensure_ascii=False),
                "dataset_layer": "atomic_positive",
                "gold_edit_count": 1,
                "count_toward_rule_quota": True,
            },
            {
                "rule_ids": json.dumps(["unit_real"], ensure_ascii=False),
                "dataset_layer": "real_atomic",
                "gold_edit_count": 1,
                "count_toward_rule_quota": True,
            },
            {
                "rule_ids": json.dumps(["unit_multi"], ensure_ascii=False),
                "dataset_layer": "atomic_positive",
                "gold_edit_count": 2,
                "count_toward_rule_quota": True,
            },
            {
                "rule_ids": json.dumps(["unit_stress_a", "unit_stress_b"], ensure_ascii=False),
                "dataset_layer": "stress_multi_error",
                "gold_edit_count": 2,
                "count_toward_rule_quota": False,
            },
            {
                "rule_ids": json.dumps(["unit_hard"], ensure_ascii=False),
                "dataset_layer": "atomic_hard_negative",
                "gold_edit_count": 0,
                "count_toward_rule_quota": False,
            },
        ]
    )

    assert _operator_rule_counts(frame) == {"unit_atomic": 1}


def test_destructive_diversity_pruning_disabled_by_default(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "source": "В тексте есть млоко.",
                "target": "В тексте есть молоко.",
                "source_type": SYNTHETIC_OPEN_CLEAN,
                "rule_id": "unit_atomic",
                "rule_ids": json.dumps(["unit_atomic"], ensure_ascii=False),
                "dataset_layer": "atomic_positive",
                "count_toward_rule_quota": True,
                "gold_edit_count": 1,
            }
        ]
    )

    def fake_audit(_frame, active_rule_ids):
        return {
            "rule_diversity_summary": {
                "failed_rule_count": 1,
                "failed_rule_ids": list(active_rule_ids),
            }
        }

    import src.data.operator_dataset_builder as operator_builder

    monkeypatch.setattr(operator_builder, "audit_training_dataset", fake_audit)

    result_frame, active_rule_ids, excluded_rule_ids, audit = _prune_failed_diversity_rules(
        frame=frame,
        active_rule_ids=["unit_atomic"],
        excluded_rule_ids={},
        requested_total=1,
        split_sizes={"train": 1, "val": 0, "test": 0},
        seed=17,
        config={},
    )

    assert len(result_frame) == 1
    assert active_rule_ids == ["unit_atomic"]
    assert excluded_rule_ids == {}
    assert "rule_diversity_warning:unit_atomic" in audit["warnings"]


def test_destructive_diversity_pruning_requires_explicit_config(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "source": "В тексте есть млоко.",
                "target": "В тексте есть молоко.",
                "source_type": SYNTHETIC_OPEN_CLEAN,
                "rule_id": "unit_atomic",
                "rule_ids": json.dumps(["unit_atomic"], ensure_ascii=False),
                "dataset_layer": "atomic_positive",
                "count_toward_rule_quota": True,
                "gold_edit_count": 1,
            }
        ]
    )

    def fake_audit(_frame, active_rule_ids):
        return {
            "rule_diversity_summary": {
                "failed_rule_count": len(active_rule_ids),
                "failed_rule_ids": list(active_rule_ids),
            }
        }

    import src.data.operator_dataset_builder as operator_builder

    monkeypatch.setattr(operator_builder, "audit_training_dataset", fake_audit)

    result_frame, active_rule_ids, excluded_rule_ids, _audit = _prune_failed_diversity_rules(
        frame=frame,
        active_rule_ids=["unit_atomic"],
        excluded_rule_ids={},
        requested_total=1,
        split_sizes={"train": 1, "val": 0, "test": 0},
        seed=17,
        config={"data": {"audit": {"destructive_diversity_pruning_enabled": True}}},
    )

    assert not result_frame["rule_ids"].astype(str).str.contains("unit_atomic", regex=False).any()
    assert active_rule_ids == []
    assert excluded_rule_ids == {"unit_atomic": "diversity_failed"}


def test_final_active_gates_use_pre_gate_counts_and_warn_below_target():
    final_active, under_quota, blockers, warnings, audit_errors = _final_active_rule_ids_after_gates(
        expanded_candidate_rule_ids=["unit_atomic"],
        provisional_active_rule_ids=["unit_atomic"],
        pre_gate_atomic_counts_by_rule={"unit_atomic": 500},
        pre_gate_hard_negative_counts_by_rule={"unit_atomic": 200},
        pre_gate_candidate_recall_by_rule={"unit_atomic": 1.0},
        pre_gate_gold_counts_by_rule={"unit_atomic": 1},
        quota_config={
            "min_atomic_positives_per_active_rule": 500,
            "preferred_atomic_positives_per_active_rule": 1500,
            "min_hard_negatives_per_active_rule": 200,
        },
        config={
            "data": {
                "audit": {"min_candidate_recall_for_active_rule": 0.95},
                "rule_activation": {
                    "target_final_active_rule_count": 76,
                    "warn_below_target_final_active_rule_count": True,
                    "expected_min_final_active_rule_count": 25,
                    "fail_below_final_active_rule_count": False,
                },
            }
        },
    )

    assert final_active == ["unit_atomic"]
    assert under_quota[0]["status"] == "active"
    assert blockers == {}
    assert "active_rule_below_preferred:unit_atomic" in warnings
    assert "final_active_rule_count_below_target:1<76" in warnings
    assert audit_errors == []


def test_final_active_gates_block_atomic_min_hard_min_and_final_min():
    rule_ids = [f"rule_{index}" for index in range(24)]
    atomic_counts = {rule_id: 500 for rule_id in rule_ids}
    hard_counts = {rule_id: 200 for rule_id in rule_ids}
    atomic_counts["atomic_low"] = 499
    hard_counts["atomic_low"] = 200
    atomic_counts["hard_low"] = 500
    hard_counts["hard_low"] = 199

    final_active, under_quota, blockers, warnings, audit_errors = _final_active_rule_ids_after_gates(
        expanded_candidate_rule_ids=[*rule_ids, "atomic_low", "hard_low"],
        provisional_active_rule_ids=[*rule_ids, "atomic_low", "hard_low"],
        pre_gate_atomic_counts_by_rule=atomic_counts,
        pre_gate_hard_negative_counts_by_rule=hard_counts,
        pre_gate_candidate_recall_by_rule={rule_id: 1.0 for rule_id in [*rule_ids, "atomic_low", "hard_low"]},
        pre_gate_gold_counts_by_rule={rule_id: 1 for rule_id in [*rule_ids, "atomic_low", "hard_low"]},
        quota_config={
            "min_atomic_positives_per_active_rule": 500,
            "preferred_atomic_positives_per_active_rule": 1500,
            "min_hard_negatives_per_active_rule": 200,
        },
        config={
            "data": {
                "audit": {"min_candidate_recall_for_active_rule": 0.95},
                "rule_activation": {
                    "expected_min_final_active_rule_count": 25,
                    "fail_below_final_active_rule_count": True,
                    "target_final_active_rule_count": 76,
                    "warn_below_target_final_active_rule_count": True,
                },
            }
        },
    )

    under_by_rule = {row["rule_id"]: row for row in under_quota}
    assert final_active == sorted(rule_ids)
    assert under_by_rule["atomic_low"]["status"] == "under_quota"
    assert under_by_rule["atomic_low"]["reason"] == "below_min_atomic_positive_quota"
    assert under_by_rule["hard_low"]["status"] == "under_quota"
    assert under_by_rule["hard_low"]["reason"] == "below_min_hard_negative_quota"
    assert blockers["atomic_low"] == "below_min_atomic_positive_quota"
    assert blockers["hard_low"] == "below_min_hard_negative_quota"
    assert "final_active_rule_count_below_target:24<76" in warnings
    assert "final_active_rule_count_below_min:24<25" in audit_errors


def test_pre_gate_count_helpers_accept_rows_and_frames():
    rows = [
        {
            "rule_ids": json.dumps(["unit_atomic"], ensure_ascii=False),
            "dataset_layer": "atomic_positive",
            "count_toward_rule_quota": True,
            "gold_edit_count": 1,
        },
        {
            "target_rule_id": "unit_atomic",
            "metadata": json.dumps({"target_rule_id": "unit_atomic"}, ensure_ascii=False),
            "dataset_layer": "atomic_hard_negative",
        },
    ]

    assert _pre_gate_atomic_counts_by_rule(rows) == {"unit_atomic": 1}
    assert _pre_gate_atomic_counts_by_rule(pd.DataFrame(rows)) == {"unit_atomic": 1}
    assert _pre_gate_hard_negative_counts_by_rule(rows) == {"unit_atomic": 1}
    assert _pre_gate_hard_negative_counts_by_rule(pd.DataFrame(rows)) == {"unit_atomic": 1}


def test_final_targeted_fill_is_capped_by_fallback_share_budget():
    rows = []
    for idx in range(100):
        rows.append(
            {
                "source_type": SYNTHETIC_OPEN_CLEAN,
                "metadata": json.dumps({"error_bearing_sentence_source": "corpus"}),
            }
        )
    for idx in range(20):
        rows.append(
            {
                "source_type": SYNTHETIC_OPEN_CLEAN,
                "metadata": json.dumps({"error_bearing_sentence_source": "fallback_template"}),
            }
        )

    assert _remaining_fallback_template_budget(rows, max_share=0.20) == 5


def test_quality_source_minimum_targets_do_not_require_exact_synthetic_target():
    targets = _quality_source_minimum_targets(
        {
            SYNTHETIC_OPEN_CLEAN: 180365,
            REAL_ERROR_PAIR: 1969,
            CLEAN_IDENTITY_OPEN: 25035,
            HARD_NEGATIVE_OPEN: 25035,
        },
        232404,
    )

    assert SYNTHETIC_OPEN_CLEAN not in targets
    assert targets[REAL_ERROR_PAIR] == 1969
    assert targets[CLEAN_IDENTITY_OPEN] == 23241
    assert targets[HARD_NEGATIVE_OPEN] == 23241


def test_dataset_build_workers_default_to_serial_and_clamp_config(monkeypatch):
    monkeypatch.delenv("RUSSIAN_CORRECTOR_DATASET_WORKERS", raising=False)

    assert _dataset_build_workers({"data": {}}) == 1
    assert _dataset_build_workers({"data": {"dataset_build_workers": 0}}) == 1
    assert _dataset_build_workers({"data": {"dataset_build_workers": 9999}}) == max(1, os.cpu_count() or 1)


def test_parallel_atomic_positive_merge_is_deterministic_and_strips_worker_keys():
    quota_config = {
        "min_atomic_positives_per_active_rule": 1,
        "preferred_atomic_positives_per_active_rule": 1,
        "max_total_per_rule_id": 1,
    }
    chunk_results = [
        {
            "rows": [
                {
                    "source": "late source",
                    "target": "late target",
                    "rule_id": "rule_a",
                    "_parallel_pool_index": 9,
                    "_parallel_rule_order": 0,
                    "_parallel_local_index": 0,
                },
                {
                    "source": "rule b source",
                    "target": "rule b target",
                    "rule_id": "rule_b",
                    "_parallel_pool_index": 1,
                    "_parallel_rule_order": 1,
                    "_parallel_local_index": 0,
                },
            ],
            "rejections": [],
            "attempts": {"rule_a": 2, "rule_b": 1},
            "opportunities_seen": {"rule_a": 2, "rule_b": 1},
            "rejection_counts": {},
        },
        {
            "rows": [
                {
                    "source": "early source",
                    "target": "early target",
                    "rule_id": "rule_a",
                    "_parallel_pool_index": 0,
                    "_parallel_rule_order": 0,
                    "_parallel_local_index": 0,
                }
            ],
            "rejections": [],
            "attempts": {"rule_a": 1},
            "opportunities_seen": {"rule_a": 1},
            "rejection_counts": {},
        },
    ]

    result, seen_hashes, counts = _merge_parallel_atomic_positive_results(
        rule_ids=["rule_a", "rule_b"],
        chunk_results=chunk_results,
        target_per_rule=1,
        quota_config=quota_config,
    )

    assert _chunk_rule_ids_for_workers(["rule_a", "rule_b", "rule_c"], 2) == [["rule_a", "rule_c"], ["rule_b"]]
    assert [row["source"] for row in result.rows] == ["early source", "rule b source"]
    assert all(not key.startswith("_parallel_") for row in result.rows for key in row)
    assert counts == {"rule_a": 1, "rule_b": 1}
    assert len(seen_hashes) == 2
    assert {row["rule_id"]: row["status"] for row in result.generation_rows} == {"rule_a": "ready", "rule_b": "ready"}
