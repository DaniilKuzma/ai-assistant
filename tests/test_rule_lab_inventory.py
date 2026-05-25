from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.rules.capabilities import active_rule_ids_for_training, activation_policy_from_config, load_rule_capabilities


def _write_recipes(path: Path, rules: dict) -> Path:
    path.write_text(yaml.safe_dump({"rules": rules}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _config(recipe_path: Path) -> dict:
    return {
        "data": {
            "candidate_opportunity": {
                "paths": {"rule_lab_recipes_config": str(recipe_path)},
                "rule_lab": {"enabled": True},
                "rule_data_compiler": {"enabled": True, "real_pattern_paths": []},
            }
        }
    }


def test_disabled_recipes_are_not_enabled_rule_lab_recipe_ids(tmp_path: Path):
    from src.data.rule_lab_generation import (
        all_rule_lab_recipe_rule_ids,
        disabled_rule_lab_recipe_rule_ids,
        enabled_rule_lab_recipe_rule_ids,
        rule_lab_recipe_rule_ids,
    )

    recipe_path = _write_recipes(
        tmp_path / "rule_lab_recipes.yaml",
        {
            "enabled_rule": {"enabled": True, "family": "unit", "positive_templates": [], "hard_negative_templates": []},
            "disabled_rule": {
                "enabled": False,
                "disabled_reason": "candidate_missing",
                "family": "unit",
                "positive_templates": [],
                "hard_negative_templates": [],
            },
        },
    )

    config = _config(recipe_path)

    assert all_rule_lab_recipe_rule_ids(config) == {"enabled_rule", "disabled_rule"}
    assert enabled_rule_lab_recipe_rule_ids(config) == {"enabled_rule"}
    assert disabled_rule_lab_recipe_rule_ids(config) == {"disabled_rule"}
    assert rule_lab_recipe_rule_ids(config) == {"enabled_rule"}


def test_supported_rule_ids_do_not_treat_disabled_only_recipe_as_generation_support(tmp_path: Path):
    from src.data.rule_data_compiler import _supported_rule_ids

    recipe_path = _write_recipes(
        tmp_path / "rule_lab_recipes.yaml",
        {
            "disabled_only_rule": {
                "enabled": False,
                "disabled_reason": "validator_missing",
                "family": "unit",
                "positive_templates": [],
                "hard_negative_templates": [],
            }
        },
    )

    assert _supported_rule_ids(["disabled_only_rule"], _config(recipe_path)) == []


def test_inventory_rows_include_missing_and_disabled_action_mappings(tmp_path: Path):
    from src.data.rule_data_compiler import build_rule_lab_inventory_rows

    recipe_path = _write_recipes(
        tmp_path / "rule_lab_recipes.yaml",
        {
            "candidate_blocked": {
                "enabled": False,
                "disabled_reason": "candidate_missing",
                "family": "unit",
                "positive_templates": [],
                "hard_negative_templates": [],
            },
            "validator_blocked": {
                "enabled": False,
                "disabled_reason": "validator_missing",
                "family": "unit",
                "positive_templates": [],
                "hard_negative_templates": [],
            },
            "unsafe_blocked": {
                "enabled": False,
                "disabled_reason": "unsafe_template",
                "family": "unit",
                "positive_templates": [],
                "hard_negative_templates": [],
            },
        },
    )

    rows = build_rule_lab_inventory_rows(
        ["candidate_blocked", "validator_blocked", "unsafe_blocked", "missing_rule"],
        _config(recipe_path),
    )
    by_rule = {row["rule_id"]: row for row in rows}

    assert by_rule["candidate_blocked"]["status"] == "blocked_needs_candidate_generator"
    assert by_rule["candidate_blocked"]["recommended_next_action"] == "fix_candidate_generator_or_rule_mapping"
    assert by_rule["validator_blocked"]["status"] == "blocked_needs_validator"
    assert by_rule["validator_blocked"]["recommended_next_action"] == "fix_strict_validator_or_atomic_verifier_support"
    assert by_rule["unsafe_blocked"]["status"] == "blocked_unsafe_template"
    assert by_rule["unsafe_blocked"]["recommended_next_action"] == "design_safe_target_first_templates"
    assert by_rule["missing_rule"]["status"] == "missing_rule_lab_recipe_and_no_known_miner"
    assert by_rule["missing_rule"]["recommended_next_action"] == "add_dedicated_miner_or_rule_lab_recipe"


def test_all_training_candidate_rules_appear_in_rule_lab_inventory_report(tmp_path: Path):
    from src.data.rule_data_compiler import build_rule_lab_inventory_rows, write_rule_lab_inventory_reports

    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    capabilities = load_rule_capabilities("configs/rules.yaml", config=config)
    training_rule_ids = active_rule_ids_for_training(capabilities, policy=activation_policy_from_config(config))

    rows = build_rule_lab_inventory_rows(training_rule_ids, config, capabilities=capabilities)
    write_rule_lab_inventory_reports(rows, tmp_path)

    report = pd.read_csv(tmp_path / "rule_lab_inventory_report.csv")
    assert set(report["rule_id"]) == set(training_rule_ids)
    assert len(report) == len(training_rule_ids) == 76


def test_inventory_script_writes_reports_without_dataset_outputs(tmp_path: Path):
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    candidate = config["data"]["candidate_opportunity"]
    candidate["paths"]["processed_dir"] = str(processed_dir)
    candidate["paths"]["clean_pool_path"] = str(processed_dir / "missing_clean_pool.csv.gz")
    candidate["paths"]["correction_dataset_path"] = str(processed_dir / "correction_dataset.csv.gz")
    candidate["paths"]["manifest_path"] = str(processed_dir / "dataset_manifest.json")
    candidate["paths"]["reports_dir"] = str(reports_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_rule_lab_inventory.py",
            "--config",
            str(config_path),
            "--reports-dir",
            str(reports_dir),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(result.stdout)

    assert summary["training_candidate_rule_count"] == 76
    for name in (
        "rule_lab_inventory_report.csv",
        "rule_lab_missing_recipe_report.csv",
        "rule_lab_disabled_recipe_report.csv",
        "rule_lab_generation_support_matrix.csv",
    ):
        assert (reports_dir / name).exists()
    assert not (processed_dir / "correction_dataset.csv.gz").exists()
    assert not (processed_dir / "train.csv").exists()
    assert not (processed_dir / "val.csv").exists()
    assert not (processed_dir / "test.csv").exists()
