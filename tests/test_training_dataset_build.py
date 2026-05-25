from pathlib import Path

import json
import pandas as pd
import yaml

from src.candidates.candidate_generator import Candidate
from src.config.candidate_dataset_config import candidate_dataset_paths, candidate_dataset_totals
import src.data.operator_dataset_builder as operator_builder
from src.data.full_dataset_builder import build_dataset_from_config
from src.rules.capabilities import RuleCapability
from tests.candidate_contract_fixtures import (
    UnitCandidateGenerator,
    candidate_contract_config,
    patch_unit_operator_pipeline,
    unit_capability,
    write_unit_clean_pool,
    write_unit_real_outputs,
)


def _candidate(config: dict) -> dict:
    return config["data"]["candidate_opportunity"]


def _dataset_path(config: dict) -> Path:
    return Path(candidate_dataset_paths(config)["correction_dataset_path"])


def _manifest_path(config: dict) -> Path:
    return Path(candidate_dataset_paths(config)["manifest_path"])


def _write_limited_unit_clean_pool(path: Path, count: int) -> Path:
    labels = ["сегодня", "утром", "вечером", "завтра", "позже"]
    rows = [
        {
            "text": f"Редакция отметила, что молоко поступило после проверки {labels[index]}.",
            "source_name": "unit",
            "source_subcorpus": "unit",
            "domain": "unit",
            "sentence_id": f"limited-{index}",
            "hash": f"limited-{index}",
        }
        for index in range(count)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class UnitSyntaxCandidateGenerator(UnitCandidateGenerator):
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


class UnitSyntaxCandidateGeneratorFactory:
    @classmethod
    def from_config(cls, config: dict):
        del config
        return UnitSyntaxCandidateGenerator()


def _syntax_capability() -> RuleCapability:
    return RuleCapability(
        taxonomy_key="unit_syntax",
        domain="punctuation",
        entry_type="rule",
        title="unit syntax",
        orfogrammka_id="",
        project_rule_ids=["comma_subordinate"],
        implementation_status="model_required",
        requires=["syntax", "model", "validator"],
        executable=True,
        training_eligible=True,
        training_decision="INCLUDE_AFTER_THRESHOLD_CALIBRATION",
        training_reason="unit syntax",
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


def _patch_unit_syntax_pipeline(monkeypatch) -> None:
    import src.data.operator_dataset_builder as operator_builder
    import src.rules.syntax_synthetic as syntax_module

    monkeypatch.setattr(operator_builder, "CandidateGenerator", UnitSyntaxCandidateGeneratorFactory, raising=False)
    monkeypatch.setattr(operator_builder, "load_rule_capabilities", lambda _path, **_kwargs: [unit_capability(), _syntax_capability()])
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


def test_canonical_config_uses_broad_dataset_targets_and_no_versioned_paths():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    paths = candidate_dataset_paths(config)
    totals = candidate_dataset_totals(config)

    assert set(config["data"]) == {"candidate_opportunity"}
    assert paths["correction_dataset_path"] == "data/processed/correction_dataset.csv.gz"
    assert paths["manifest_path"] == "data/processed/dataset_manifest.json"
    assert totals["total_examples"] >= 200000
    assert totals["train_examples"] == int(totals["total_examples"] * 0.8)
    assert totals["val_examples"] == int(totals["total_examples"] * 0.1)
    assert totals["test_examples"] == (
        totals["total_examples"] - totals["train_examples"] - totals["val_examples"]
    )
    assert config["training"]["max_train_examples"] == totals["train_examples"]
    assert config["training"]["max_val_examples"] == totals["val_examples"]
    assert config["training"]["max_test_examples"] == totals["test_examples"]

    serialized = yaml.safe_dump(config["data"], allow_unicode=True)
    forbidden = ("short_dataset_v2", "short_dataset_v3", "current_capability_v", "wave", "phase", "latest")
    assert not any(marker in serialized for marker in forbidden)


def test_candidate_contract_pipeline_builds_atomic_from_clean_pool_without_output_seed(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder

    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    output_path = _dataset_path(config)
    patch_unit_operator_pipeline(monkeypatch)

    def fail_if_output_is_read(path: Path):
        raise AssertionError(f"output dataset must not be read as seed: {path}")

    monkeypatch.setattr(operator_builder, "_read_seed_dataset", fail_if_output_is_read, raising=False)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    frame = pd.read_csv(output_path)
    atomic = frame[frame["rule_id"].astype(str).eq("unit_atomic")]

    assert result["total"] == 10
    assert result["dataset_contract"] == "candidate_opportunity"
    assert len(result["dataset_hash"]) == 64
    assert result["audit_errors"] == manifest["audit_errors"]
    assert result["layer_counts"] == manifest["layer_counts"]
    assert output_path.exists()
    assert frame["split"].value_counts().to_dict() == {"train": 8, "val": 1, "test": 1}
    assert manifest["dataset_contract"] == "candidate_opportunity"
    assert len(manifest["dataset_hash"]) == 64
    assert len(manifest["config_hash"]) == 64
    assert manifest["layer_counts"]["atomic_positive"] == 2
    assert manifest["rule_id_counts_atomic_positive_only"]["unit_atomic"] == 2
    assert manifest["hard_negative_counts_by_target_rule"]["unit_atomic"] >= 1
    assert len(atomic) == 2
    assert atomic.iloc[0]["source"] != atomic.iloc[0]["target"]
    assert atomic.iloc[0]["dataset_contract"] == "candidate_opportunity"
    assert atomic.iloc[0]["dataset_layer"] == "atomic_positive"
    assert bool(atomic.iloc[0]["count_toward_rule_quota"]) is True
    assert int(atomic.iloc[0]["gold_edit_count"]) == 1
    assert not (frame["dataset_layer"].eq("atomic_positive") & frame["rule_id"].eq("clean_identity")).any()
    assert (output_path.parent / "train_atomic_positive.csv.gz").exists()
    assert (output_path.parent / "train_atomic_hard_negative.csv.gz").exists()
    assert (output_path.parent / "train_clean_identity.csv.gz").exists()
    assert (output_path.parent / "train_real_atomic.csv.gz").exists()
    assert (output_path.parent / "train_stress_multi_error.csv.gz").exists()
    quota = pd.read_csv(Path(config["paths"]["reports_dir"]) / "dataset_build" / "active_rule_quota_report.csv")
    assert list(quota.columns)[:8] == [
        "rule_id",
        "pre_gate_atomic_positive_count",
        "selected_atomic_positive_count",
        "pre_gate_hard_negative_count",
        "selected_hard_negative_count",
        "min_atomic_required",
        "preferred_atomic",
        "min_hard_negative_required",
    ]
    assert list(quota.columns)[8:11] == [
        "candidate_recall",
        "status",
        "reason",
    ]
    assert quota.set_index("rule_id").loc["unit_atomic", "pre_gate_atomic_positive_count"] == 2
    assert quota.set_index("rule_id").loc["unit_atomic", "selected_atomic_positive_count"] == 2
    assert quota.set_index("rule_id").loc["unit_atomic", "min_hard_negative_required"] == 1
    assert {
        "operator_atomic_count",
        "syntax_synthetic_atomic_count",
        "real_atomic_count",
        "total_atomic_positive_count",
    } <= set(quota.columns)
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    assert manifest["expanded_training_candidate_rule_ids"] == ["unit_atomic"]
    assert manifest["expanded_training_candidate_rule_count"] == 1
    assert manifest["final_active_rule_ids"] == ["unit_atomic"]
    assert manifest["final_active_rule_count"] == 1
    assert manifest["active_rule_ids"] == manifest["final_active_rule_ids"]
    assert manifest["training_candidate_rule_ids"] == manifest["expanded_training_candidate_rule_ids"]
    assert manifest["under_quota_rule_ids"] == []
    assert (reports_dir / "expanded_activation_candidate_report.csv").exists()
    assert (reports_dir / "expanded_activation_blocked_report.csv").exists()
    assert (reports_dir / "active_rule_activation_stage_report.csv").exists()
    assert (reports_dir / "under_quota_active_rules_report.csv").exists()
    assert (reports_dir / "layer_target_report.csv").exists()
    assert (reports_dir / "active_rule_coverage_report.csv").exists()
    assert (reports_dir / "hard_negative_coverage_report.csv").exists()
    assert manifest["requested_layer_targets"]["atomic_positive"] == 4
    assert manifest["effective_layer_targets"]["atomic_positive"] == 2
    assert manifest["layer_target_adjustments"]["atomic_positive"]["reason"] == "requested_exceeds_per_rule_capacity"
    assert "atomic_positive_target_adjusted" in manifest["warnings"]
    layer_report = pd.read_csv(reports_dir / "layer_target_report.csv").set_index("layer")
    assert int(layer_report.loc["atomic_positive", "requested_target"]) == 4
    assert int(layer_report.loc["atomic_positive", "effective_target"]) == 2
    assert int(layer_report.loc["atomic_positive", "pre_gate_available_count"]) == 2
    assert int(layer_report.loc["atomic_positive", "selected_count"]) == 2
    assert int(layer_report.loc["atomic_positive", "removed_by_final_active_filter"]) == 0
    assert int(layer_report.loc["atomic_positive", "removed_by_diversity_pruning"]) == 0
    assert int(layer_report.loc["atomic_positive", "removed_by_quality_sanitizer"]) == 0
    assert int(layer_report.loc["atomic_positive", "deficit"]) == 2
    assert layer_report.loc["atomic_positive", "adjustment_reason"] == "requested_exceeds_per_rule_capacity"
    active_coverage = pd.read_csv(reports_dir / "active_rule_coverage_report.csv").set_index("rule_id")
    assert {"pre_gate_atomic_positive_count", "selected_atomic_positive_count"} <= set(active_coverage.columns)
    assert active_coverage.loc["unit_atomic", "status"] == "pass"
    assert int(active_coverage.loc["unit_atomic", "pre_gate_atomic_positive_count"]) == 2
    assert int(active_coverage.loc["unit_atomic", "selected_atomic_positive_count"]) == 2
    hard_coverage = pd.read_csv(reports_dir / "hard_negative_coverage_report.csv").set_index("target_rule_id")
    assert hard_coverage.loc["unit_atomic", "status"] == "pass"
    candidate_report = pd.read_csv(reports_dir / "expanded_activation_candidate_report.csv").set_index("rule_id")
    assert candidate_report.loc["unit_atomic", "included"] in (True, "True", "true")
    assert candidate_report.loc["unit_atomic", "verifier_pass_count"] == 2
    assert candidate_report.loc["unit_atomic", "candidate_recall"] == 1.0


def test_candidate_contract_pipeline_reports_under_quota_rules_without_training_rows(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    _candidate(config)["rule_quota"]["min_hard_negatives_per_active_rule"] = 5
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    hard_coverage = pd.read_csv(reports_dir / "hard_negative_coverage_report.csv").set_index("target_rule_id")
    guard = pd.read_csv(reports_dir / "atomic_positive_drop_guard_report.csv").iloc[0]

    assert result["verdict"] == "DATASET_BLOCKED"
    assert "unit_atomic" in manifest["expanded_training_candidate_rule_ids"]
    assert "unit_atomic" in manifest["under_quota_rule_ids"]
    assert "unit_atomic" not in manifest["final_active_rule_ids"]
    assert "hard_negative_under_min:unit_atomic" in manifest["audit_errors"]
    assert "destructive_atomic_positive_drop_detected" in manifest["audit_errors"]
    assert manifest["rules_under_hard_negative_min"] == ["unit_atomic"]
    assert hard_coverage.loc["unit_atomic", "status"] == "fail"
    assert hard_coverage.loc["unit_atomic", "reason"] == "hard_negative_under_min"
    assert int(guard["pre_gate_atomic_positive_available_count"]) > 0
    assert int(guard["selected_atomic_positive_count"]) == 0
    assert _dataset_path(config).exists() is False


def test_candidate_contract_pipeline_blocks_silent_atomic_positive_drop(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    patch_unit_operator_pipeline(monkeypatch)

    def drop_all_final_active_rules(**_kwargs):
        return [], [], {"unit_atomic": "forced_empty_final_active"}, [], []

    monkeypatch.setattr(operator_builder, "_final_active_rule_ids_after_gates", drop_all_final_active_rules)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    guard = pd.read_csv(reports_dir / "atomic_positive_drop_guard_report.csv").iloc[0]

    assert result["verdict"] == "DATASET_BLOCKED"
    assert "destructive_atomic_positive_drop_detected" in result["audit_errors"]
    assert manifest["final_active_rule_count"] == 0
    assert int(guard["pre_gate_atomic_positive_available_count"]) > 0
    assert int(guard["selected_atomic_positive_count"]) == 0
    assert int(guard["removed_by_final_active_filter"]) > 0
    assert guard["suspected_reason"] == "final_active_filter"
    assert not _dataset_path(config).exists()


def test_candidate_contract_pipeline_counts_syntax_synthetic_rows_toward_quota(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    candidate = _candidate(config)
    candidate["rule_quota"]["rule_ids"] = ["unit_atomic", "comma_subordinate"]
    candidate["composition"]["atomic_positive_target"] = 4
    candidate["composition"]["atomic_hard_negative_target"] = 4
    candidate["composition"]["clean_identity_target"] = 2
    patch_unit_operator_pipeline(monkeypatch)
    _patch_unit_syntax_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    frame = pd.read_csv(_dataset_path(config))
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    quota = pd.read_csv(reports_dir / "active_rule_quota_report.csv").set_index("rule_id")

    syntax_atomic = frame[
        frame["dataset_layer"].eq("atomic_positive")
        & frame["rule_id"].eq("comma_subordinate")
    ]
    syntax_hard = frame[
        frame["dataset_layer"].eq("atomic_hard_negative")
        & frame["target_rule_id"].eq("comma_subordinate")
    ]

    assert result["dataset_contract"] == "candidate_opportunity"
    assert len(syntax_atomic) == 2
    assert syntax_atomic["activation_source"].eq("syntax_synthetic").all()
    assert syntax_atomic["count_toward_rule_quota"].astype(bool).all()
    assert int(quota.loc["comma_subordinate", "syntax_synthetic_atomic_count"]) == 2
    assert int(quota.loc["comma_subordinate", "operator_atomic_count"]) == 0
    assert int(quota.loc["comma_subordinate", "total_atomic_positive_count"]) == 2
    assert int(manifest["syntax_atomic_positive_count_by_rule"]["comma_subordinate"]) == 2
    assert "comma_subordinate" in manifest["syntax_supported_training_candidate_rule_ids"]
    assert "comma_subordinate" in manifest["syntax_supported_active_rule_ids"]
    assert (reports_dir / "syntax_atomic_positive_generation_report.csv").exists()
    assert (reports_dir / "syntax_atomic_positive_rejection_report.csv").exists()
    assert (reports_dir / "syntax_active_rule_coverage_report.csv").exists()
    assert (reports_dir / "syntax_hard_negative_coverage_report.csv").exists()
    assert not syntax_hard["count_toward_rule_quota"].astype(bool).any()


def test_candidate_contract_pipeline_smoke_links_real_outputs_and_prefers_composition(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder

    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    real_paths = write_unit_real_outputs(_dataset_path(config).parent)
    candidate = _candidate(config)
    candidate["totals"] = {
        "total_examples": 8,
        "train_examples": 6,
        "val_examples": 1,
        "test_examples": 1,
    }
    candidate["paths"]["real_error_pairs_atomic_path"] = str(real_paths["atomic"])
    candidate["paths"]["real_error_pairs_stress_path"] = str(real_paths["stress"])
    candidate["paths"]["real_error_pairs_validated_path"] = str(real_paths["validated"])
    candidate["composition"] = {
        "atomic_positive_target": 2,
        "atomic_hard_negative_target": 2,
        "clean_identity_target": 2,
        "real_atomic_train_target": 1,
        "stress_multi_error_target": 1,
    }
    patch_unit_operator_pipeline(monkeypatch)

    def fail_if_output_is_read(path: Path):
        raise AssertionError(f"output dataset must not be read as seed: {path}")

    monkeypatch.setattr(operator_builder, "_read_seed_dataset", fail_if_output_is_read, raising=False)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    frame = pd.read_csv(_dataset_path(config))
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"

    assert result["total"] == 8
    assert result["dataset_hash"] == manifest["dataset_hash"]
    assert manifest["layer_counts"] == {
        "atomic_positive": 2,
        "atomic_hard_negative": 2,
        "clean_identity": 2,
        "real_atomic": 1,
        "stress_multi_error": 1,
    }
    atomic = frame[frame["dataset_layer"].eq("atomic_positive")]
    real_atomic = frame[frame["dataset_layer"].eq("real_atomic")]
    stress = frame[frame["dataset_layer"].eq("stress_multi_error")]
    assert atomic["gold_edit_count"].astype(int).tolist() == [1, 1]
    assert atomic["count_toward_rule_quota"].astype(bool).all()
    assert real_atomic["gold_edit_count"].astype(int).tolist() == [1]
    assert not real_atomic["count_toward_rule_quota"].astype(bool).any()
    assert stress["gold_edit_count"].astype(int).tolist() == [2]
    assert not stress["count_toward_rule_quota"].astype(bool).any()
    assert stress["loss_weight"].astype(float).tolist() == [0.4]
    assert Path(real_paths["mining"]).exists()
    assert not frame["dataset_layer"].eq("real_mining").any()
    assert (reports_dir / "active_rule_quota_report.csv").exists()
    assert (reports_dir / "candidate_recall_by_rule.csv").exists()
    assert (reports_dir / "candidate_recall_gate_report.csv").exists()
    assert (reports_dir / "atomic_purity_report.csv").exists()
    assert (reports_dir / "real_pair_atomization_report.csv").exists()


def test_candidate_contract_pipeline_cleans_reports_and_blocks_stale_report_hashes(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    reports_dir.mkdir(parents=True)
    stale_path = reports_dir / "old_false_report.csv"
    stale_path.write_text("stale,report\n1,1\n", encoding="utf-8")
    patch_unit_operator_pipeline(monkeypatch)

    build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    generation_report = reports_dir / "dataset_generation_report.md"
    generation_text = generation_report.read_text(encoding="utf-8")

    assert not stale_path.exists()
    assert f"- dataset_hash: {manifest['dataset_hash']}" in generation_text
    assert f"- config_hash: {manifest['config_hash']}" in generation_text
    assert (reports_dir / "report_manifest.json").exists()

    candidate_report = reports_dir / "candidate_recall_by_rule.csv"
    candidate_report.write_text(
        candidate_report.read_text(encoding="utf-8").replace("1.0", "0.0", 1),
        encoding="utf-8",
    )

    result = build_dataset_from_config(config, force=False)
    blocked_manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["verdict"] == "DATASET_BLOCKED"
    assert "stale_reports_hash_mismatch:candidate_recall_by_rule.csv" in result["audit_errors"]
    assert blocked_manifest["report_freshness"]["status"] == "stale"
    assert "stale_reports_hash_mismatch:candidate_recall_by_rule.csv" in blocked_manifest["report_freshness"]["errors"]


def test_candidate_contract_pipeline_reads_canonical_audit(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    _candidate(config)["audit"]["min_candidate_recall_for_active_rule"] = 1.01
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))

    assert result["verdict"] == "DATASET_BLOCKED"
    assert "candidate_recall_active_min_below_threshold" in result["audit_errors"]
    assert "candidate_recall_under_min:unit_atomic" in result["audit_errors"]
    assert manifest["audit_errors"] == result["audit_errors"]


def test_candidate_contract_pipeline_passes_canonical_stress_loss_weight(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder

    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    _candidate(config)["composition"]["stress_multi_error_target"] = 1
    _candidate(config)["stress"]["loss_weight"] = 0.23
    patch_unit_operator_pipeline(monkeypatch)
    captured: dict[str, float] = {}

    class EmptyStressResult:
        rows = []
        attempt_rows = []
        rejection_rows = []
        counts_by_rule_combo = {}

    def fake_generate_multi_error_stress_rows(*args, **kwargs):
        del args
        captured["loss_weight"] = kwargs["loss_weight"]
        return EmptyStressResult()

    monkeypatch.setattr(operator_builder, "generate_multi_error_stress_rows", fake_generate_multi_error_stress_rows)

    build_dataset_from_config(config, force=True)

    assert captured["loss_weight"] == 0.23


def test_build_dataset_script_summary_payload_includes_contract_hash_audits_and_layers():
    from scripts.build_dataset import _dataset_summary_payload

    payload = _dataset_summary_payload(
        {
            "dataset_contract": "candidate_opportunity",
            "dataset_hash": "a" * 64,
            "verdict": "READY_FOR_TRAINING_DATASET",
            "audit_errors": [],
            "layer_counts": {"atomic_positive": 2},
        },
        manifest_path=None,
    )

    assert payload == {
        "dataset_contract": "candidate_opportunity",
        "dataset_hash": "a" * 64,
        "verdict": "READY_FOR_TRAINING_DATASET",
        "audit_errors": [],
        "layer_counts": {"atomic_positive": 2},
    }


def test_candidate_contract_pipeline_blocks_when_clean_pool_missing(tmp_path: Path, monkeypatch):
    patch_unit_operator_pipeline(monkeypatch)
    config = candidate_contract_config(tmp_path, tmp_path / "missing_clean_sentence_pool.csv.gz")
    output_path = _dataset_path(config)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["verdict"] == "DATASET_BLOCKED"
    assert manifest["audit_errors"] == ["missing_clean_sentence_pool"]
    assert not output_path.exists()


def test_candidate_contract_pipeline_blocks_atomic_positive_under_min(tmp_path: Path, monkeypatch):
    clean_pool_path = _write_limited_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz", 2)
    config = candidate_contract_config(tmp_path, clean_pool_path)
    candidate = _candidate(config)
    candidate["totals"] = {"total_examples": 4, "train_examples": 2, "val_examples": 1, "test_examples": 1}
    candidate["composition"]["atomic_positive_target"] = 3
    candidate["composition"]["atomic_hard_negative_target"] = 1
    candidate["composition"]["clean_identity_target"] = 0
    candidate["rule_quota"]["min_atomic_positives_per_active_rule"] = 3
    candidate["rule_quota"]["preferred_atomic_positives_per_active_rule"] = 3
    candidate["rule_quota"]["max_total_per_rule_id"] = 3
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    active_coverage = pd.read_csv(reports_dir / "active_rule_coverage_report.csv").set_index("rule_id")

    assert result["verdict"] == "DATASET_BLOCKED"
    assert "atomic_positive_under_min:unit_atomic" in manifest["audit_errors"]
    assert active_coverage.loc["unit_atomic", "status"] == "fail"
    assert active_coverage.loc["unit_atomic", "reason"] == "atomic_positive_under_min"


def test_candidate_contract_pipeline_warns_atomic_below_preferred_without_block(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    _candidate(config)["rule_quota"]["min_atomic_positives_per_active_rule"] = 1
    _candidate(config)["rule_quota"]["preferred_atomic_positives_per_active_rule"] = 3
    _candidate(config)["rule_quota"]["max_total_per_rule_id"] = 2
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))
    reports_dir = Path(config["paths"]["reports_dir"]) / "dataset_build"
    active_coverage = pd.read_csv(reports_dir / "active_rule_coverage_report.csv").set_index("rule_id")

    assert result["verdict"] == "READY_FOR_TRAINING_DATASET"
    assert manifest["audit_errors"] == []
    assert "active_rule_below_preferred:unit_atomic" in manifest["warnings"]
    assert manifest["active_rules_below_preferred"] == ["unit_atomic"]
    assert active_coverage.loc["unit_atomic", "status"] == "warning"
    assert active_coverage.loc["unit_atomic", "reason"] == "active_rule_below_preferred"


def test_candidate_contract_pipeline_final_active_count_gate_blocks(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    _candidate(config)["rule_activation"]["expected_min_final_active_rule_count"] = 25
    _candidate(config)["rule_activation"]["fail_below_final_active_rule_count"] = True
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))

    assert result["verdict"] == "DATASET_BLOCKED"
    assert manifest["final_active_rule_count"] == 1
    assert "final_active_rule_count_below_min:1<25" in manifest["audit_errors"]


def test_candidate_contract_pipeline_final_active_target_warning_does_not_block(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    _candidate(config)["rule_activation"]["expected_min_final_active_rule_count"] = 0
    _candidate(config)["rule_activation"]["fail_below_final_active_rule_count"] = False
    _candidate(config)["rule_activation"]["target_final_active_rule_count"] = 76
    _candidate(config)["rule_activation"]["warn_below_target_final_active_rule_count"] = True
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(_manifest_path(config).read_text(encoding="utf-8"))

    assert result["verdict"] == "READY_FOR_TRAINING_DATASET"
    assert manifest["final_active_rule_count"] == 1
    assert "final_active_rule_count_below_target:1<76" in manifest["warnings"]
    assert not any(error.startswith("final_active_rule_count_below_min:") for error in manifest["audit_errors"])


def test_post_compose_sanitizer_removes_bad_clean_and_hard_rows():
    clean_bad = operator_builder._identity_row(
        'Он сказал: "текст готов.',
        operator_builder.CLEAN_IDENTITY_OPEN,
        "unit",
        "unit",
        "open_clean",
    )
    hard_bad = operator_builder._identity_row(
        "Он сделал так же (как раньше.",
        operator_builder.HARD_NEGATIVE_OPEN,
        "unit",
        "unit",
        "open_clean",
    )
    atomic = {
        "source": "В отчете было млоко.",
        "target": "В отчете было молоко.",
        "split": "train",
        "source_type": operator_builder.SYNTHETIC_OPEN_CLEAN,
        "dataset_layer": operator_builder.LAYER_ATOMIC_POSITIVE,
        "rule_id": "unit_atomic",
        "rule_ids": json.dumps(["unit_atomic"], ensure_ascii=False),
        "metadata": "{}",
        "normalized_pair_hash": "atomic",
        "gold_edit_count": 1,
    }
    clean_good = operator_builder._identity_row(
        "Команда проверила короткую заметку утром после заседания редакции.",
        operator_builder.CLEAN_IDENTITY_OPEN,
        "unit",
        "unit",
        "open_clean",
    )
    hard_good = operator_builder._identity_row(
        "Он сделал так же, как раньше.",
        operator_builder.HARD_NEGATIVE_OPEN,
        "unit",
        "unit",
        "open_clean",
    )

    result = operator_builder._post_compose_sanitize_clean_hard_rows(
        pd.DataFrame([clean_bad, hard_bad, atomic]),
        clean_identity_pool=[clean_good],
        hard_negative_pool=[hard_good],
        requested_total=3,
        split_sizes={"train": 3, "val": 0, "test": 0},
        seed=1,
    )

    assert len(result.frame) == 3
    assert result.audit_errors == []
    assert result.removed_by_layer == {
        operator_builder.LAYER_ATOMIC_POSITIVE: 0,
        operator_builder.LAYER_ATOMIC_HARD_NEGATIVE: 1,
        operator_builder.LAYER_CLEAN_IDENTITY: 1,
        operator_builder.LAYER_REAL_ATOMIC: 0,
        operator_builder.LAYER_STRESS_MULTI_ERROR: 0,
    }
    reasons = {row["reason"] for row in result.removed_rows}
    assert "unbalanced_ascii_quotes" in reasons
    assert "unbalanced_parentheses" in reasons
    assert not result.frame["source"].astype(str).str.contains('"текст готов|\\(как раньше', regex=True).any()
