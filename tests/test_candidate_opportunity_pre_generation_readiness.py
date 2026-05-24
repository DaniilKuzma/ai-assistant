from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.data.full_dataset_builder import build_dataset_from_config
from src.data.training_quality_audit import report_manifest_errors
from src.rules.capabilities import RuleCapability
from tests.candidate_contract_fixtures import (
    UnitAtomicOperator,
    UnitCandidateGenerator,
    candidate_contract_config,
    unit_capability,
    write_unit_clean_pool,
    write_unit_real_outputs,
)


class ReadinessCandidateGenerator(UnitCandidateGenerator):
    def generate(self, text: str):
        candidates = list(super().generate(text))
        marker = "готов"
        index = text.find(marker)
        if index >= 0:
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


class ReadinessCandidateGeneratorFactory:
    @classmethod
    def from_config(cls, config: dict):
        del config
        return ReadinessCandidateGenerator()


def _syntax_capability() -> RuleCapability:
    return RuleCapability(
        taxonomy_key="fixture_comma_subordinate",
        domain="punctuation",
        entry_type="rule",
        title="comma subordinate fixture",
        orfogrammka_id="",
        project_rule_ids=["comma_subordinate"],
        implementation_status="model_required",
        requires=["syntax", "model", "validator"],
        executable=True,
        training_eligible=True,
        training_decision="INCLUDE_AFTER_THRESHOLD_CALIBRATION",
        training_reason="tiny syntax fixture",
        has_candidate_path=True,
        has_synthetic_support=True,
        has_hard_negative_support=True,
        has_validator_support=True,
        has_dictionary_support=False,
        has_syntax_support=True,
        has_morphology_support=False,
        has_ner_support=False,
        risk_level="medium",
    )


def _blocked_capability() -> RuleCapability:
    return RuleCapability(
        taxonomy_key="fixture_blocked",
        domain="test",
        entry_type="rule",
        title="blocked fixture",
        orfogrammka_id="",
        project_rule_ids=["blocked_unit"],
        implementation_status="blocked",
        requires=[],
        executable=False,
        training_eligible=False,
        training_decision="BLOCK_DISABLED",
        training_reason="blocked fixture",
        has_candidate_path=True,
        has_synthetic_support=True,
        has_hard_negative_support=True,
        has_validator_support=True,
        has_dictionary_support=False,
        has_syntax_support=False,
        has_morphology_support=False,
        has_ner_support=False,
        risk_level="low",
    )


def _patch_readiness_pipeline(monkeypatch) -> None:
    import src.data.operator_dataset_builder as operator_builder
    import src.rules.syntax_synthetic as syntax_module
    from src.data.corruption_operators import RuleOperatorRegistry

    registry = RuleOperatorRegistry()
    registry.register(UnitAtomicOperator())
    capabilities = [unit_capability(), _syntax_capability(), _blocked_capability()]
    monkeypatch.setattr(operator_builder, "build_default_operator_registry", lambda: registry)
    monkeypatch.setattr(operator_builder, "CandidateGenerator", ReadinessCandidateGeneratorFactory, raising=False)
    monkeypatch.setattr(operator_builder, "load_rule_capabilities", lambda _path, **_kwargs: capabilities)
    monkeypatch.setattr(syntax_module, "SUPPORTED_SYNTAX_RULE_IDS", ("comma_subordinate",))
    monkeypatch.setattr(
        syntax_module,
        "build_syntax_eval_examples",
        lambda **_kwargs: pd.DataFrame(
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
                    "metadata": json.dumps({"candidate_present": True}, ensure_ascii=False),
                },
                {
                    "source": "Если архив готов мы обновим журнал вечером.",
                    "target": "Если архив готов, мы обновим журнал вечером.",
                    "rule_id": "comma_subordinate",
                    "syntax_family": "subordinate_clause_comma",
                    "source_type": "syntax_synthetic_eval",
                    "candidate_present": True,
                    "candidate_rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
                    "hard_negative": False,
                    "metadata": json.dumps({"candidate_present": True}, ensure_ascii=False),
                },
            ]
        ),
    )
    monkeypatch.setattr(
        syntax_module,
        "build_syntax_hard_negatives",
        lambda **_kwargs: pd.DataFrame(
            [
                {
                    "source": "Когда отчет готов мы отправим письмо утром.",
                    "target": "Когда отчет готов мы отправим письмо утром.",
                    "rule_id": "comma_subordinate",
                    "syntax_family": "subordinate_clause_comma",
                    "source_type": "syntax_hard_negative_trap_candidate",
                    "candidate_present": True,
                    "candidate_rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
                    "hard_negative": True,
                    "metadata": json.dumps({"candidate_present": True}, ensure_ascii=False),
                    "hard_negative_kind": "trap_candidate",
                }
            ]
        ),
    )


def _readiness_config(tmp_path: Path) -> tuple[dict, Path]:
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    real_paths = write_unit_real_outputs(Path(config["data"]["processed_train_path"]).parent)
    config["data"]["total_examples"] = 12
    config["data"]["target_total_examples"] = 12
    config["data"]["train_examples"] = 10
    config["data"]["val_examples"] = 1
    config["data"]["test_examples"] = 1
    config["data"]["exact_split_sizes"] = {"train": 10, "val": 1, "test": 1}
    config["data"]["real_error_pairs_atomic_path"] = str(real_paths["atomic"])
    config["data"]["real_error_pairs_stress_path"] = str(real_paths["stress"])
    config["data"]["real_error_pairs_validated_path"] = str(real_paths["validated"])
    config["data"]["composition"] = {
        "atomic_positive_target": 4,
        "atomic_hard_negative_target": 4,
        "clean_identity_target": 2,
        "real_atomic_train_target": 1,
        "stress_multi_error_target": 1,
    }
    config["data"]["rule_activation"].update(
        {
            "mode": "expanded_safe",
            "expected_min_production_ready_rule_count": 1,
            "expected_min_training_candidate_rule_count": 2,
            "expected_min_final_active_rule_count": 2,
            "target_training_candidate_rule_count": 2,
            "target_final_active_rule_count": 2,
            "fail_below_min_training_candidate_rule_count": True,
            "warn_below_target_training_candidate_rule_count": True,
            "fail_below_final_active_rule_count": True,
            "warn_below_target_final_active_rule_count": True,
        }
    )
    config["data"]["rule_quota"] = {
        "rule_ids": ["unit_atomic", "comma_subordinate", "blocked_unit"],
        "min_atomic_positives_per_active_rule": 1,
        "preferred_atomic_positives_per_active_rule": 2,
        "max_total_per_rule_id": 2,
        "min_hard_negatives_per_active_rule": 1,
        "disable_rule_if_quota_not_met": True,
    }
    config["data"]["audit"] = {"min_candidate_recall_for_active_rule": 0.95}
    conflicting_legacy_targets = {
        "synthetic_augmented_from_open_clean": 12,
        "real_error_pair": 0,
        "clean_identity_from_open_clean": 0,
        "hard_negative_from_open_clean": 0,
    }
    for section in ("training_dataset", "training_dataset_core"):
        config["data"][section]["legacy_builder"] = False
        config["data"][section]["source_type_targets"] = dict(conflicting_legacy_targets)
        config["data"][section]["split_source_type_targets"] = {
            "train": dict(conflicting_legacy_targets),
            "val": {},
            "test": {},
        }
    config["data"]["training_dataset_core"]["active_rule_quota"] = {
        "rule_ids": ["unit_atomic", "comma_subordinate", "blocked_unit"],
        "min_total_per_active_rule": 1,
        "preferred_total_per_active_rule": 2,
        "split_minimums": {},
    }
    config["data"]["training_dataset_core"]["rule_caps"]["max_total_per_rule_id"] = 2
    config["data"]["training_dataset_core"]["audit"]["require_all_source_types"] = False
    return config, real_paths["mining"]


def _assert_fresh_report_manifest(reports_dir: Path, manifest: dict) -> None:
    report_manifest_path = reports_dir / "report_manifest.json"
    assert report_manifest_path.exists()
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    assert report_manifest["dataset_hash"] == manifest["dataset_hash"]
    assert report_manifest["config_hash"] == manifest["config_hash"]
    assert report_manifest_errors(
        reports_dir,
        dataset_hash=manifest["dataset_hash"],
        config_hash=manifest["config_hash"],
    ) == []


def test_tiny_candidate_opportunity_pre_generation_readiness(tmp_path: Path, monkeypatch):
    _patch_readiness_pipeline(monkeypatch)
    config, mining_path = _readiness_config(tmp_path)
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    reports_dir.mkdir(parents=True)
    stale_path = reports_dir / "old_report_without_current_hashes.csv"
    stale_path.write_text("stale,report\n1,1\n", encoding="utf-8")

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(config["data"]["processed_train_path"])

    assert result["verdict"] == "READY_FOR_TRAINING_DATASET"
    assert result["dataset_contract"] == "candidate_opportunity"
    assert manifest["dataset_contract"] == "candidate_opportunity"
    assert len(manifest["dataset_hash"]) == 64
    assert len(manifest["config_hash"]) == 64
    assert not stale_path.exists()
    assert manifest["production_ready_rule_ids"] == ["unit_atomic"]
    assert set(manifest["training_candidate_rule_ids"]) == {"unit_atomic", "comma_subordinate"}
    assert set(manifest["active_rule_ids"]) == {"unit_atomic", "comma_subordinate"}
    assert manifest["final_active_rule_count"] == 2
    assert manifest["final_active_rule_count"] > manifest["production_ready_rule_count"]
    assert manifest["target_training_candidate_rule_count"] == 2
    assert manifest["target_final_active_rule_count"] == 2
    assert "blocked_unit" in manifest["blocked_rule_ids"]
    assert "blocked_unit" not in manifest["active_rule_ids"]
    assert not frame["rule_ids"].astype(str).str.contains("blocked_unit", regex=False).any()
    assert not frame["target_rule_id"].astype(str).str.contains("blocked_unit", regex=False).any()
    assert config["data"]["training_dataset"]["legacy_builder"] is False
    assert config["data"]["training_dataset_core"]["legacy_builder"] is False
    assert manifest["layer_counts"] == {
        "atomic_positive": 4,
        "atomic_hard_negative": 4,
        "clean_identity": 2,
        "real_atomic": 1,
        "stress_multi_error": 1,
    }

    syntax_atomic = frame[
        frame["dataset_layer"].eq("atomic_positive")
        & frame["activation_source"].eq("syntax_synthetic")
    ]
    atomic = frame[frame["dataset_layer"].eq("atomic_positive")]
    hard = frame[frame["dataset_layer"].eq("atomic_hard_negative")]
    stress = frame[frame["dataset_layer"].eq("stress_multi_error")]
    assert len(syntax_atomic) > 0
    assert int(manifest["syntax_atomic_positive_count"]) == len(syntax_atomic)
    assert atomic["gold_edit_count"].astype(int).eq(1).all()
    assert hard["source"].eq(hard["target"]).all()
    assert not hard["count_toward_rule_quota"].astype(bool).any()
    assert stress["gold_edit_count"].astype(int).ge(2).all()
    assert not stress["count_toward_rule_quota"].astype(bool).any()
    assert mining_path.exists()

    for name in ("layer_target_report.csv", "active_rule_coverage_report.csv", "hard_negative_coverage_report.csv"):
        assert (reports_dir / name).exists()
    quota = pd.read_csv(reports_dir / "active_rule_quota_report.csv").set_index("rule_id")
    assert quota.loc["unit_atomic", "status"] == "ready"
    assert quota.loc["comma_subordinate", "status"] == "ready"
    assert int(quota.loc["comma_subordinate", "syntax_synthetic_atomic_count"]) > 0
    active_coverage = pd.read_csv(reports_dir / "active_rule_coverage_report.csv").set_index("rule_id")
    hard_coverage = pd.read_csv(reports_dir / "hard_negative_coverage_report.csv").set_index("target_rule_id")
    assert active_coverage.loc["unit_atomic", "status"] in {"pass", "warning"}
    assert active_coverage.loc["comma_subordinate", "status"] in {"pass", "warning"}
    assert hard_coverage.loc["unit_atomic", "status"] == "pass"
    assert hard_coverage.loc["comma_subordinate", "status"] == "pass"
    generation_report = (reports_dir / "dataset_generation_report.md").read_text(encoding="utf-8")
    assert f"- dataset_hash: {manifest['dataset_hash']}" in generation_report
    assert f"- config_hash: {manifest['config_hash']}" in generation_report
    _assert_fresh_report_manifest(reports_dir, manifest)

    candidate_report = reports_dir / "candidate_recall_by_rule.csv"
    candidate_report.write_text(
        candidate_report.read_text(encoding="utf-8").replace("1.0", "0.0", 1),
        encoding="utf-8",
    )
    stale_result = build_dataset_from_config(config, force=False)
    blocked_manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
    assert stale_result["status"] == "blocked"
    assert stale_result["verdict"] == "DATASET_BLOCKED"
    assert "stale_reports_hash_mismatch:candidate_recall_by_rule.csv" in stale_result["audit_errors"]
    assert blocked_manifest["report_freshness"]["status"] == "stale"
