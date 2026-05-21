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
    canonical_config = config["data"]["training_dataset"]

    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
    assert config["data"]["target_total_examples"] == 100000
    assert config["data"]["train_examples"] == 80000
    assert config["data"]["val_examples"] == 10000
    assert config["data"]["test_examples"] == 10000
    assert "fallback_split_sizes" not in canonical_config
    assert canonical_config["smoke"]["enabled"] is False
    assert canonical_config["smoke"]["verdict"] == "READY_FOR_SMOKE_ONLY"


def test_canonical_100k_exact_sizes():
    config = _load_canonical_config()
    data = config["data"]
    targets = data["training_dataset"]["source_type_targets"]
    split_targets = data["training_dataset"]["split_source_type_targets"]

    assert data["target_total_examples"] == 100000
    assert data["exact_split_sizes"] == {"train": 80000, "val": 10000, "test": 10000}
    assert targets == {
        "synthetic_augmented_from_open_clean": 72000,
        "real_error_pair": 4000,
        "clean_identity_from_open_clean": 12000,
        "hard_negative_from_open_clean": 12000,
    }
    assert split_targets["train"] == {
        "synthetic_augmented_from_open_clean": 57600,
        "real_error_pair": 3200,
        "clean_identity_from_open_clean": 9600,
        "hard_negative_from_open_clean": 9600,
    }
    assert split_targets["val"] == {
        "synthetic_augmented_from_open_clean": 7200,
        "real_error_pair": 400,
        "clean_identity_from_open_clean": 1200,
        "hard_negative_from_open_clean": 1200,
    }
    assert split_targets["test"] == split_targets["val"]


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
    assert result["verdict"] == "BLOCKED"
    assert manifest["verdict"] == "BLOCKED"
    assert manifest["fallback_used"] is False
    assert manifest["requested_total"] == 100000
    assert manifest["actual_total"] == 48687


def test_canonical_preserves_stable_core_rules():
    rows = training.resolve_active_target_rules(_load_canonical_config())
    active = {row["rule_id"]: row for row in rows if row["include_in_dataset"]}
    core_stable = {
        row["rule_id"]
        for row in rows
        if row["tier"] == "stable_core" and row["source"] == "core" and float(row["candidate_recall"]) >= 0.85
    }

    assert core_stable
    assert core_stable <= set(active)
    assert all(active[rule_id]["quota_min"] == 300 for rule_id in core_stable - training.RISKY_LEXICAL_RULE_IDS)
    assert all(active[rule_id]["quota_preferred"] == 700 for rule_id in core_stable - training.RISKY_LEXICAL_RULE_IDS)


def test_canonical_includes_activation_rules():
    rows = training.resolve_active_target_rules(_load_canonical_config())
    by_rule = {row["rule_id"]: row for row in rows}

    assert ACTIVATION_INCLUDED_RULES <= {rule_id for rule_id, row in by_rule.items() if row["include_in_dataset"]}
    assert all(by_rule[rule_id]["tier"] in {"activation", "stable_core"} for rule_id in ACTIVATION_INCLUDED_RULES)
    assert all(by_rule[rule_id]["quota_min"] in {300, 500} for rule_id in ACTIVATION_INCLUDED_RULES)
    assert all(by_rule[rule_id]["quota_preferred"] in {700, 1000} for rule_id in ACTIVATION_INCLUDED_RULES)
    if "hyphen_whitelist" in by_rule:
        assert by_rule["hyphen_whitelist"]["include_in_dataset"] is False
        assert by_rule["hyphen_whitelist"]["reason"] == "under_quota_nonblocking"


def test_canonical_synthetic_min_70000():
    config = _load_canonical_config()
    canonical_config = config["data"]["training_dataset"]

    assert canonical_config["source_type_targets"]["synthetic_augmented_from_open_clean"] == 72000
    assert canonical_config["audit"]["synthetic_min"] == 70000


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
        assert config["paths"]["adapter_output_dir"] == "models/current/adapters"
        assert config["paths"]["heads_output_dir"] == "models/current/heads"

    assert e1["training"]["epochs"] == 1
