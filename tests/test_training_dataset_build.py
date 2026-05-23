from pathlib import Path

import json
import pandas as pd
import yaml

from src.data.full_dataset_builder import build_dataset_from_config
from tests.candidate_contract_fixtures import (
    UnitCandidateGenerator,
    candidate_contract_config,
    patch_unit_operator_pipeline,
    write_unit_clean_pool,
)


def test_canonical_config_uses_broad_dataset_targets_and_no_versioned_paths():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
    assert config["data"]["total_examples"] >= 200000
    assert config["data"]["train_examples"] == int(config["data"]["total_examples"] * 0.8)
    assert config["data"]["val_examples"] == int(config["data"]["total_examples"] * 0.1)
    assert config["data"]["test_examples"] == (
        config["data"]["total_examples"] - config["data"]["train_examples"] - config["data"]["val_examples"]
    )
    assert config["training"]["max_train_examples"] == config["data"]["train_examples"]
    assert config["training"]["max_val_examples"] == config["data"]["val_examples"]
    assert config["training"]["max_test_examples"] == config["data"]["test_examples"]

    serialized = yaml.safe_dump(config["data"], allow_unicode=True)
    forbidden = ("short_dataset_v2", "short_dataset_v3", "current_capability_v", "wave", "phase", "latest")
    assert not any(marker in serialized for marker in forbidden)


def test_candidate_contract_pipeline_builds_atomic_from_clean_pool_without_output_seed(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder

    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    output_path = Path(config["data"]["processed_train_path"])
    patch_unit_operator_pipeline(monkeypatch)

    def fail_if_output_is_read(path: Path):
        raise AssertionError(f"output dataset must not be read as seed: {path}")

    monkeypatch.setattr(operator_builder, "_read_seed_dataset", fail_if_output_is_read, raising=False)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
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
    assert list(quota.columns) == [
        "rule_id",
        "atomic_positive_count",
        "hard_negative_count",
        "min_required",
        "hard_negative_min_required",
        "preferred",
        "status",
        "reason",
    ]
    assert quota.set_index("rule_id").loc["unit_atomic", "atomic_positive_count"] == 2
    assert quota.set_index("rule_id").loc["unit_atomic", "hard_negative_min_required"] == 1


def test_candidate_contract_pipeline_reads_top_level_audit_alias_first(tmp_path: Path, monkeypatch):
    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    config["data"]["audit"] = {"min_candidate_recall_for_active_rule": 1.01}
    config["data"]["training_dataset_core"]["audit"]["candidate_recall_min"] = 0.0
    patch_unit_operator_pipeline(monkeypatch)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))

    assert result["verdict"] == "DATASET_BLOCKED"
    assert "candidate_recall_active_min_below_threshold" in result["audit_errors"]
    assert manifest["audit_errors"] == result["audit_errors"]


def test_candidate_contract_pipeline_passes_top_level_stress_loss_weight(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder

    clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = candidate_contract_config(tmp_path, clean_pool_path)
    config["data"]["composition"]["stress_multi_error_target"] = 1
    config["data"]["stress"]["loss_weight"] = 0.23
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
    output_path = Path(config["data"]["processed_train_path"])

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["verdict"] == "DATASET_BLOCKED"
    assert manifest["audit_errors"] == ["missing_clean_sentence_pool"]
    assert not output_path.exists()
