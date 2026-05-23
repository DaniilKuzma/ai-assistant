from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data._training_dataset_builder import _targeted_fill_rule_for_attempt
import src.data.training_dataset as training


ACTIVATION_INCLUDED_RULES = {
    "address_comma",
    "comma_conjunction",
    "comma_subordinate",
    "comparative_turnover_comma",
    "detached_adverbial_comma",
    "direct_speech_dash",
    "homogeneous_comma",
    "hyphen_koe_koy",
    "hyphen_particles",
    "introductory_comma",
    "subject_predicate_dash",
}


def _load_canonical_config() -> dict:
    return yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))


def test_training_dataset_config_is_canonical_single_artifact():
    config = _load_canonical_config()
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
    canonical_config = config["data"]["training_dataset"]
    total = manifest["total"]
    split_sizes = manifest["split_sizes"]

    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
    assert config["data"]["target_total_examples"] == total
    assert config["data"]["total_examples"] == total
    assert total >= 200000
    assert config["data"]["train_examples"] == split_sizes["train"]
    assert config["data"]["val_examples"] == split_sizes["val"]
    assert config["data"]["test_examples"] == split_sizes["test"]
    assert config["training"]["max_train_examples"] == split_sizes["train"]
    assert config["training"]["max_val_examples"] == split_sizes["val"]
    assert config["training"]["max_test_examples"] == split_sizes["test"]
    assert split_sizes["train"] == int(total * 0.8)
    assert split_sizes["val"] == int(total * 0.1)
    assert split_sizes["test"] == total - split_sizes["train"] - split_sizes["val"]
    assert "fallback_split_sizes" not in canonical_config
    assert canonical_config["smoke"]["enabled"] is False
    assert canonical_config["smoke"]["verdict"] == "READY_FOR_SMOKE_ONLY"


def test_canonical_broad_exact_sizes():
    config = _load_canonical_config()
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
    data = config["data"]
    targets = data["training_dataset"]["source_type_targets"]
    split_targets = data["training_dataset"]["split_source_type_targets"]
    split_sizes = manifest["split_sizes"]
    composition = manifest["composition"]

    assert data["target_total_examples"] == manifest["total"]
    assert data["exact_split_sizes"] == split_sizes
    assert sum(targets.values()) == manifest["total"]
    assert targets["synthetic_augmented_from_open_clean"] == composition["synthetic_augmented_from_open_clean"]
    assert targets["real_error_pair"] == composition["real_error_pair"]
    assert targets["clean_identity_from_open_clean"] == composition["clean_identity_from_open_clean"]
    assert targets["hard_negative_from_open_clean"] == composition["hard_negative_from_open_clean"]
    assert targets["synthetic_augmented_from_open_clean"] >= data["training_dataset"]["audit"]["synthetic_min"]
    assert targets["real_error_pair"] > 0

    for split, split_source_targets in split_targets.items():
        assert sum(split_source_targets.values()) == split_sizes[split]
    for source_type, target_count in targets.items():
        assert sum(split_targets[split][source_type] for split in ("train", "val", "test")) == target_count


def test_canonical_no_60k_ready_fallback(monkeypatch, tmp_path: Path):
    config = deepcopy(_load_canonical_config())
    manifest_path = tmp_path / "dataset_manifest.json"
    config["data"]["processed_train_path"] = str(tmp_path / "correction_dataset.csv.gz")
    config["data"]["manifest_path"] = str(manifest_path)
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    calls = []

    def fake_core_builder(builder_config: dict, force: bool = False) -> dict:
        calls.append(builder_config)
        manifest_path.write_text(
            json.dumps(
                {
                    "total": 48687,
                    "split_sizes": {"train": 40572, "val": 4057, "test": 4058},
                    "composition": {"synthetic_augmented_from_open_clean": 30687},
                    "candidate_recall_summary": {"active_min_excluding_unknown": 1.0},
                    "gap_label_coverage_summary": {"active_min_excluding_unknown": 1.0},
                    "active_rule_ids": [],
                    "excluded_active_rule_ids": [],
                    "rule_id_counts": {},
                    "rule_id_counts_by_split": {"train": {}, "val": {}, "test": {}},
                    "low_count_active_rule_ids": [],
                    "verdict": "BLOCKED",
                    "audit_errors": ["dataset_size_below_requested:48687!=100000"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"status": "blocked", "verdict": "BLOCKED", "total": 48687, "manifest_path": str(manifest_path)}

    monkeypatch.setattr(training, "build_training_dataset_core_from_config", fake_core_builder)

    result = training.build_training_dataset_from_config(config, force=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert result["verdict"] == "DATASET_BLOCKED"
    assert manifest["verdict"] == "DATASET_BLOCKED"
    assert manifest["fallback_used"] is False
    assert manifest["requested_total"] == config["data"]["target_total_examples"]
    assert manifest["actual_total"] == 48687


def test_canonical_resolves_broad_active_rules():
    rows = training.resolve_active_target_rules(_load_canonical_config())
    active = {row["rule_id"]: row for row in rows if row["include_in_dataset"]}

    assert len(active) == 76
    assert active["comma_subordinate"]["quota_min"] == 1500
    assert active["dictionary_fuzzy"]["quota_preferred"] == 1800
    assert active["final_punctuation_default"]["quota_preferred"] == 3000


def test_canonical_includes_activation_rules():
    rows = training.resolve_active_target_rules(_load_canonical_config())
    by_rule = {row["rule_id"]: row for row in rows}

    assert ACTIVATION_INCLUDED_RULES <= {rule_id for rule_id, row in by_rule.items() if row["include_in_dataset"]}
    assert all(by_rule[rule_id]["tier"] in {"syntax_supported", "legacy_stable", "current_capability"} for rule_id in ACTIVATION_INCLUDED_RULES)
    assert all(by_rule[rule_id]["quota_min"] >= 1000 for rule_id in ACTIVATION_INCLUDED_RULES)
    assert all(by_rule[rule_id]["quota_preferred"] >= 2500 for rule_id in ACTIVATION_INCLUDED_RULES)
    if "hyphen_whitelist" in by_rule:
        assert by_rule["hyphen_whitelist"]["include_in_dataset"] is True
        assert by_rule["hyphen_whitelist"]["reason"] == "legacy_candidate_backed_current_capability"


def test_canonical_synthetic_min_70000():
    config = _load_canonical_config()
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
    canonical_config = config["data"]["training_dataset"]
    synthetic_target = canonical_config["source_type_targets"]["synthetic_augmented_from_open_clean"]

    assert synthetic_target == manifest["composition"]["synthetic_augmented_from_open_clean"]
    assert synthetic_target >= canonical_config["audit"]["synthetic_min"]


def test_canonical_candidate_recall_active_only():
    report = pd.DataFrame(
        [
            {"rule_id": "active_rule", "gold_count": 10, "candidate_recall": 0.91},
            {"rule_id": "excluded_rule", "gold_count": 10, "candidate_recall": 0.10},
        ]
    )

    summary = training.active_metric_summary(
        report,
        count_column="gold_count",
        metric_column="candidate_recall",
        active_rule_ids={"active_rule"},
    )

    assert summary["active_min_excluding_unknown"] == 0.91
    assert summary["active_mean_excluding_unknown"] == 0.91


def test_canonical_excluded_rules_not_counted_underfilled():
    active_rows = [
        {"rule_id": "active_rule", "include_in_dataset": True, "quota_min": 500},
        {"rule_id": "excluded_rule", "include_in_dataset": False, "quota_min": 500},
    ]

    underfilled = training.underfilled_active_rule_ids(active_rows, {"active_rule": 500, "excluded_rule": 0})

    assert underfilled == []


def test_canonical_targeted_fill_rotates_past_lowest_unfillable_rule():
    ordered_rules = ["unfillable_low", "fillable_mid", "fillable_high"]
    counts = {"unfillable_low": 10, "fillable_mid": 20, "fillable_high": 30}

    selected = [
        _targeted_fill_rule_for_attempt(ordered_rules, counts, attempt_index=index)
        for index in range(3)
    ]

    assert selected == ["unfillable_low", "fillable_mid", "fillable_high"]


def test_train_training_config_uses_canonical_outputs():
    e1 = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    e2 = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    for config in (e1, e2):
        assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
        assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
        assert config["thresholds"]["mode"] == "calibrated_guarded"
        assert config["model"]["max_sequence_length"] == 128
        assert config["model"]["max_candidates"] == 16
        assert config["paths"]["adapter_output_dir"] == "models/adapters/latest"
        assert config["paths"]["heads_output_dir"] == "models/heads/latest"

    assert e1["training"]["epochs"] == 1
