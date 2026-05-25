from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import yaml

from src.config.candidate_dataset_config import (
    candidate_dataset_paths,
    candidate_dataset_rule_activation,
    candidate_dataset_rule_quota,
    candidate_dataset_totals,
    get_candidate_dataset_config,
    validate_candidate_dataset_config,
)


def _canonical_config() -> dict:
    return {
        "data": {
            "candidate_opportunity": {
                "contract": "candidate_opportunity",
                "totals": {
                    "total_examples": 240000,
                    "train_examples": 192000,
                    "val_examples": 24000,
                    "test_examples": 24000,
                },
                "paths": {
                    "processed_dir": "data/processed",
                    "reports_dir": "reports/dataset_build",
                    "clean_pool_path": "data/processed/clean_sentence_pool.csv.gz",
                    "correction_dataset_path": "data/processed/correction_dataset.csv.gz",
                    "manifest_path": "data/processed/dataset_manifest.json",
                    "open_corpora_sources_config": "configs/open_corpora_sources.yaml",
                    "real_error_sources_config": "configs/real_error_sources.yaml",
                    "rule_lab_recipes_config": "configs/rule_lab_recipes.yaml",
                },
                "rule_activation": {
                    "mode": "expanded_safe",
                    "expected_min_production_ready_rule_count": 12,
                    "expected_min_training_candidate_rule_count": 76,
                    "target_training_candidate_rule_count": 76,
                    "fail_below_min_training_candidate_rule_count": True,
                    "warn_below_target_training_candidate_rule_count": True,
                    "fail_below_final_active_rule_count": True,
                    "expected_min_final_active_rule_count": 25,
                    "target_final_active_rule_count": 76,
                    "warn_below_target_final_active_rule_count": True,
                },
                "composition": {
                    "atomic_positive_ratio": 0.50,
                    "atomic_hard_negative_ratio": 0.25,
                    "clean_identity_ratio": 0.12,
                    "stress_multi_error_ratio": 0.05,
                    "real_atomic_train_ratio": 0.03,
                    "allow_layer_target_adjustment": True,
                    "fail_on_unadjusted_layer_deficit": True,
                },
                "clean_pool": {
                    "reject_mixed_script_tokens": True,
                    "reject_latin_confusable_inside_cyrillic_word": True,
                    "reject_if_candidate_generator_finds_high_confidence_fix": True,
                    "high_confidence_candidate_threshold": 0.95,
                },
                "stress": {
                    "enabled": True,
                    "count_toward_rule_quota": False,
                    "loss_weight": 0.4,
                    "min_ratio": 0.03,
                    "max_ratio": 0.05,
                    "fail_on_under_target": False,
                },
                "rule_quota": {
                    "min_atomic_positives_per_active_rule": 500,
                    "preferred_atomic_positives_per_active_rule": 1500,
                    "max_total_per_rule_id": 1500,
                    "min_hard_negatives_per_active_rule": 200,
                    "disable_rule_if_quota_not_met": True,
                },
                "rule_data_compiler": {"enabled": True},
                "rule_lab": {"enabled": False},
                "audit": {
                    "fail_on_extra_edits_in_atomic": True,
                    "fail_on_unknown_rule_in_train": True,
                    "fail_on_stale_reports": True,
                    "min_candidate_recall_for_active_rule": 0.95,
                    "fail_on_corpus_opportunity_share_below_threshold": False,
                    "destructive_diversity_pruning_enabled": False,
                },
            }
        }
    }


def _main_config() -> dict:
    return yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))


def test_current_config_uses_only_candidate_opportunity_dataset_block():
    data = _main_config()["data"]

    assert "candidate_opportunity" in data
    assert "training_dataset" not in data
    assert "training_dataset_core" not in data
    for duplicate in (
        "dataset_contract",
        "target_total_examples",
        "total_examples",
        "train_examples",
        "val_examples",
        "test_examples",
        "exact_split_sizes",
        "composition",
        "clean_pool",
        "synthetic",
        "real_pairs",
        "stress",
        "rule_quota",
        "rule_activation",
        "audit",
        "rule_data_compiler",
        "rule_lab",
    ):
        assert duplicate not in data


def test_validator_rejects_legacy_blocks_and_duplicates():
    config = _canonical_config()
    config["data"]["training_dataset"] = {"enabled": True}
    config["data"]["training_dataset_core"] = {"enabled": True}
    config["data"]["rule_quota"] = {"min_atomic_positives_per_active_rule": 1}
    config["data"]["composition"] = {"atomic_positive_ratio": 1.0}
    config["data"]["audit"] = {"fail_on_stale_reports": False}

    errors = validate_candidate_dataset_config(config)

    assert "legacy_dataset_block_present:data.training_dataset" in errors
    assert "legacy_dataset_block_present:data.training_dataset_core" in errors
    assert "duplicate_dataset_config:data.rule_quota" in errors
    assert "duplicate_dataset_config:data.composition" in errors
    assert "duplicate_dataset_config:data.audit" in errors


def test_validator_rejects_invalid_totals_and_quota_ordering():
    config = _canonical_config()
    candidate = config["data"]["candidate_opportunity"]
    candidate["totals"]["test_examples"] = 1
    candidate["rule_quota"]["min_atomic_positives_per_active_rule"] = 2000
    candidate["rule_quota"]["preferred_atomic_positives_per_active_rule"] = 1500

    errors = validate_candidate_dataset_config(config)

    assert any(error.startswith("dataset_totals_mismatch:") for error in errors)
    assert "rule_quota_min_atomic_gt_preferred_atomic:2000>1500" in errors


def test_validator_rejects_cap_activation_and_threshold_conflicts():
    config = _canonical_config()
    candidate = config["data"]["candidate_opportunity"]
    candidate["rule_quota"]["preferred_atomic_positives_per_active_rule"] = 1600
    candidate["rule_quota"]["max_total_per_rule_id"] = 1500
    candidate["rule_quota"]["min_hard_negatives_per_active_rule"] = -1
    candidate["rule_activation"]["expected_min_final_active_rule_count"] = 77
    candidate["rule_activation"]["target_training_candidate_rule_count"] = 75
    candidate["clean_pool"]["high_confidence_candidate_threshold"] = 1.1

    errors = validate_candidate_dataset_config(config)

    assert "rule_quota_preferred_atomic_gt_max_total:1600>1500" in errors
    assert "rule_quota_min_hard_negatives_negative:-1" in errors
    assert "rule_activation_expected_final_gt_target:77>76" in errors
    assert "rule_activation_target_training_below_expected_min:75<76" in errors
    assert "clean_pool_high_confidence_threshold_out_of_range:1.1" in errors


def test_resolver_returns_canonical_values_and_defaults():
    config = _canonical_config()
    audit = config["data"]["candidate_opportunity"].pop("audit")
    audit.pop("destructive_diversity_pruning_enabled")
    config["data"]["candidate_opportunity"]["audit"] = audit

    normalized = get_candidate_dataset_config(config)
    totals = candidate_dataset_totals(config)
    paths = candidate_dataset_paths(config)
    quota = candidate_dataset_rule_quota(config)
    activation = candidate_dataset_rule_activation(config)

    assert normalized["audit"]["destructive_diversity_pruning_enabled"] is False
    assert totals == {
        "total_examples": 240000,
        "train_examples": 192000,
        "val_examples": 24000,
        "test_examples": 24000,
    }
    assert quota["min_atomic_positives_per_active_rule"] == 500
    assert quota["preferred_atomic_positives_per_active_rule"] == 1500
    assert quota["min_hard_negatives_per_active_rule"] == 200
    assert activation["expected_min_final_active_rule_count"] == 25
    assert activation["target_final_active_rule_count"] == 76
    assert paths["processed_dir"] == "data/processed"
    assert paths["open_corpora_sources_config"] == "configs/open_corpora_sources.yaml"
    assert paths["real_error_sources_config"] == "configs/real_error_sources.yaml"


def test_current_config_resolver_values():
    config = _main_config()
    totals = candidate_dataset_totals(config)
    quota = candidate_dataset_rule_quota(config)
    activation = candidate_dataset_rule_activation(config)
    candidate = get_candidate_dataset_config(config)

    assert totals["total_examples"] == 240000
    assert totals["train_examples"] == 192000
    assert totals["val_examples"] == 24000
    assert totals["test_examples"] == 24000
    assert quota["min_atomic_positives_per_active_rule"] == 500
    assert quota["preferred_atomic_positives_per_active_rule"] == 1500
    assert quota["max_total_per_rule_id"] == 1500
    assert quota["min_hard_negatives_per_active_rule"] == 200
    assert activation["expected_min_final_active_rule_count"] == 25
    assert activation["target_final_active_rule_count"] == 76
    assert candidate["rule_data_compiler"]["enabled"] is True
    assert candidate["rule_lab"]["enabled"] is False


def test_production_code_does_not_read_legacy_dataset_config_directly():
    production_files = [
        Path("scripts/build_dataset.py"),
        Path("scripts/preflight_candidate_dataset.py"),
        Path("scripts/setup_data_sources.py"),
        Path("scripts/rebuild_training_dataset.py"),
        Path("src/data/operator_dataset_builder.py"),
        Path("src/data/_training_dataset_builder.py"),
        Path("src/data/full_dataset_builder.py"),
        Path("src/data/training_dataset.py"),
        Path("src/data/rule_data_compiler.py"),
        Path("src/rules/capabilities.py"),
    ]
    forbidden_keys = (
        "training_dataset",
        "training_dataset_core",
        "rule_quota",
        "rule_activation",
        "composition",
        "clean_pool",
        "stress",
        "rule_data_compiler",
        "rule_lab",
    )
    forbidden = [
        rf"config\.get\([\"']data[\"'][^;\n]*\.get\([\"']{key}[\"']" for key in forbidden_keys
    ]
    forbidden.extend(rf"\b(?:data|data_config)\.get\([\"']{key}[\"']" for key in forbidden_keys)
    forbidden.extend(rf"config\[[\"']data[\"']\]\[[\"']{key}[\"']\]" for key in forbidden_keys)
    forbidden.extend(rf"\b(?:data|data_config)\[[\"']{key}[\"']\]" for key in forbidden_keys)

    offenders: list[str] = []
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            for match in re.finditer(pattern, text):
                line_number = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path}:{line_number}:{match.group(0)}")

    assert offenders == []
