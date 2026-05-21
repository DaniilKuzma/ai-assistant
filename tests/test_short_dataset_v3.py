from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.short_dataset_v2 import _targeted_fill_rule_for_attempt
import src.data.short_dataset_v3 as short_v3


WAVE1_INCLUDED_RULES = {
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


def _load_v3_config() -> dict:
    return yaml.safe_load(Path("configs/config.short_dataset_v3.yaml").read_text(encoding="utf-8"))


def test_short_dataset_v3_config_is_canonical_single_artifact():
    config = _load_v3_config()
    v3_config = config["data"]["short_dataset_v3"]

    assert config["data"]["processed_train_path"] == "data/processed/short_dataset_v3/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "reports/short_dataset_v3/dataset_manifest.json"
    assert config["data"]["target_total_examples"] == 100000
    assert config["data"]["train_examples"] == 80000
    assert config["data"]["val_examples"] == 10000
    assert config["data"]["test_examples"] == 10000
    assert "fallback_split_sizes" not in v3_config
    assert v3_config["smoke"]["enabled"] is False
    assert v3_config["smoke"]["verdict"] == "READY_FOR_SMOKE_ONLY"


def test_v3_100k_exact_sizes():
    config = _load_v3_config()
    data = config["data"]
    targets = data["short_dataset_v3"]["source_type_targets"]
    split_targets = data["short_dataset_v3"]["split_source_type_targets"]

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


def test_v3_no_60k_ready_fallback(monkeypatch, tmp_path: Path):
    config = deepcopy(_load_v3_config())
    manifest_path = tmp_path / "dataset_manifest.json"
    config["data"]["processed_train_path"] = str(tmp_path / "correction_dataset.csv.gz")
    config["data"]["manifest_path"] = str(manifest_path)
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    calls = []

    def fake_v2_builder(builder_config: dict, force: bool = False) -> dict:
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

    monkeypatch.setattr(short_v3, "build_short_dataset_v2_from_config", fake_v2_builder)

    result = short_v3.build_short_dataset_v3_from_config(config, force=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert result["verdict"] == "BLOCKED"
    assert manifest["verdict"] == "BLOCKED"
    assert manifest["fallback_used"] is False
    assert manifest["requested_total"] == 100000
    assert manifest["actual_total"] == 48687


def test_v3_preserves_stable_v2_core_rules():
    rows = short_v3.resolve_v3_active_target_rules(_load_v3_config())
    active = {row["rule_id"]: row for row in rows if row["include_in_dataset_v3"]}
    v2_stable = {
        row["rule_id"]
        for row in rows
        if row["tier"] == "stable_v2_core" and row["source"] == "v2" and float(row["candidate_recall"]) >= 0.85
    }

    assert v2_stable
    assert v2_stable <= set(active)
    assert all(active[rule_id]["quota_min"] == 300 for rule_id in v2_stable - short_v3.RISKY_LEXICAL_RULE_IDS)
    assert all(active[rule_id]["quota_preferred"] == 700 for rule_id in v2_stable - short_v3.RISKY_LEXICAL_RULE_IDS)


def test_v3_includes_wave1_rules():
    rows = short_v3.resolve_v3_active_target_rules(_load_v3_config())
    by_rule = {row["rule_id"]: row for row in rows}

    assert WAVE1_INCLUDED_RULES <= {rule_id for rule_id, row in by_rule.items() if row["include_in_dataset_v3"]}
    assert all(by_rule[rule_id]["tier"] == "wave1" for rule_id in WAVE1_INCLUDED_RULES)
    assert all(by_rule[rule_id]["quota_min"] == 500 for rule_id in WAVE1_INCLUDED_RULES)
    assert all(by_rule[rule_id]["quota_preferred"] == 1000 for rule_id in WAVE1_INCLUDED_RULES)
    assert by_rule["hyphen_whitelist"]["include_in_dataset_v3"] is False
    assert by_rule["hyphen_whitelist"]["reason"] == "under_quota_nonblocking"


def test_v3_synthetic_min_70000():
    config = _load_v3_config()
    v3_config = config["data"]["short_dataset_v3"]

    assert v3_config["source_type_targets"]["synthetic_augmented_from_open_clean"] == 72000
    assert v3_config["audit"]["synthetic_min"] == 70000


def test_v3_candidate_recall_active_only():
    report = pd.DataFrame(
        [
            {"rule_id": "active_rule", "gold_count": 10, "candidate_recall": 0.91},
            {"rule_id": "excluded_rule", "gold_count": 10, "candidate_recall": 0.10},
        ]
    )

    summary = short_v3.active_metric_summary(
        report,
        count_column="gold_count",
        metric_column="candidate_recall",
        active_rule_ids={"active_rule"},
    )

    assert summary["active_min_excluding_unknown"] == 0.91
    assert summary["active_mean_excluding_unknown"] == 0.91


def test_v3_excluded_rules_not_counted_underfilled():
    active_rows = [
        {"rule_id": "active_rule", "include_in_dataset_v3": True, "quota_min": 500},
        {"rule_id": "excluded_rule", "include_in_dataset_v3": False, "quota_min": 500},
    ]

    underfilled = short_v3.underfilled_active_rule_ids(active_rows, {"active_rule": 500, "excluded_rule": 0})

    assert underfilled == []


def test_v3_targeted_fill_rotates_past_lowest_unfillable_rule():
    ordered_rules = ["unfillable_low", "fillable_mid", "fillable_high"]
    counts = {"unfillable_low": 10, "fillable_mid": 20, "fillable_high": 30}

    selected = [
        _targeted_fill_rule_for_attempt(ordered_rules, counts, attempt_index=index)
        for index in range(3)
    ]

    assert selected == ["unfillable_low", "fillable_mid", "fillable_high"]


def test_train_short_v3_configs_use_versioned_outputs():
    e1 = yaml.safe_load(Path("configs/config.train_short_v3_e1.yaml").read_text(encoding="utf-8"))
    e2 = yaml.safe_load(Path("configs/config.train_short_v3_e2.yaml").read_text(encoding="utf-8"))

    for config in (e1, e2):
        assert config["data"]["processed_train_path"] == "data/processed/short_dataset_v3/correction_dataset.csv.gz"
        assert config["data"]["manifest_path"] == "reports/short_dataset_v3/dataset_manifest.json"
        assert config["thresholds"]["mode"] == "calibrated_val_guarded_v3"
        assert config["model"]["max_sequence_length"] == 128
        assert config["model"]["max_candidates"] == 16
        assert "latest" not in config["paths"]["adapter_output_dir"]
        assert "latest" not in config["paths"]["heads_output_dir"]

    assert e1["training"]["epochs"] == 1
    assert e2["training"]["epochs"] == 2
