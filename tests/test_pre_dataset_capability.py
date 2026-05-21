from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml


REQUIRED_PRE_DATASET_REPORTS = {
    "candidate_path_probe.csv",
    "dataset_eligible_rules.csv",
    "dataset_blocked_rules.csv",
    "validator_needed_training_rules.csv",
    "current_capability_summary.md",
    "current_capability_dataset_plan.md",
    "rules_yaml_update_report.md",
}


def _entry(
    *,
    section: str = "orthography",
    key: str = "fixture_rule",
    title: str = "Fixture rule",
    entry_type: str = "leaf_rule",
    status: str = "model_required",
    rule_ids: list[str] | None = None,
    requires: list[str] | None = None,
    executable: bool = True,
    eligible_now: bool = False,
) -> dict[str, object]:
    return {
        "source_section": section,
        "orfogrammka_id": "fixture.1" if entry_type != "group" else None,
        "title": title,
        "normalized_title": title.lower(),
        "entry_type": entry_type,
        "parent_key": f"{section}_fixture_parent",
        "parent_path": ["Fixture"],
        "depth": 2,
        "order": 1,
        "source_url": "https://orfogrammka.ru/fixture/",
        "implementation": {
            "status": status,
            "executable": executable,
            "rule_ids": rule_ids or [],
            "aliases": [],
            "requires": requires or ["model"],
            "notes": "Fixture entry.",
        },
        "dataset": {
            "eligible_now": eligible_now,
            "reason": "fixture",
            "last_known_candidate_recall": None,
            "last_known_eval_count": None,
        },
    }


def _write_rules(path: Path, *, orthography: dict[str, object], punctuation: dict[str, object] | None = None) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "source": {
                    "taxonomy_provider": "orfogrammka",
                    "orthography_url": "https://orfogrammka.ru/орфография/",
                    "punctuation_url": "https://orfogrammka.ru/пунктуация/",
                },
                "orthography": orthography,
                "punctuation": punctuation or {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_pre_dataset_audit_blocks_metadata_planned_disabled_and_no_candidate_entries(tmp_path: Path):
    from src.evaluation.pre_dataset_capability import build_pre_dataset_capability

    rules_path = tmp_path / "rules.yaml"
    _write_rules(
        rules_path,
        orthography={
            "metadata_group": _entry(entry_type="group", status="metadata_only", executable=False, requires=[]),
            "planned_rule": _entry(status="planned", executable=False, requires=[]),
            "disabled_rule": _entry(status="disabled", executable=False, rule_ids=["yo_e_candidate"], requires=["dictionary", "model"]),
            "no_candidate_rule": _entry(status="model_required", executable=False, rule_ids=[], requires=["model"]),
        },
    )

    result = build_pre_dataset_capability(rules_config_path=rules_path, matrix_reports_dir=tmp_path / "missing")
    by_key = {row["matrix_key"]: row for row in result.entry_rows}

    assert by_key["metadata_group"]["training_eligibility_decision"] == "BLOCK_METADATA_ONLY"
    assert by_key["planned_rule"]["training_eligibility_decision"] == "BLOCK_PLANNED"
    assert by_key["disabled_rule"]["training_eligibility_decision"] == "BLOCK_DISABLED"
    assert by_key["no_candidate_rule"]["training_eligibility_decision"] == "BLOCK_NO_CANDIDATE"
    assert not any(row["training_eligible_now"] for row in by_key.values())


def test_candidate_backed_model_required_and_candidate_only_rules_can_be_training_eligible(tmp_path: Path):
    from src.evaluation.pre_dataset_capability import build_pre_dataset_capability

    rules_path = tmp_path / "rules.yaml"
    _write_rules(
        rules_path,
        orthography={
            "model_required_fixture": _entry(
                title="Final punctuation fixture",
                status="model_required",
                rule_ids=["final_punctuation_default"],
                requires=["model"],
            ),
            "candidate_only_fixture": _entry(
                title="Hyphen particles fixture",
                status="candidate_only",
                rule_ids=["hyphen_particles"],
                requires=["model"],
            ),
        },
    )

    result = build_pre_dataset_capability(rules_config_path=rules_path, matrix_reports_dir=tmp_path / "missing")
    by_key = {row["matrix_key"]: row for row in result.entry_rows}

    assert by_key["model_required_fixture"]["training_eligible_now"] is True
    assert by_key["candidate_only_fixture"]["training_eligible_now"] is True
    assert by_key["model_required_fixture"]["current_candidate_path"] is True
    assert by_key["candidate_only_fixture"]["current_candidate_path"] is True
    assert result.summary["training_eligible_active_rule_count"] >= 2


def test_pre_dataset_writer_updates_rules_yaml_and_creates_reports(tmp_path: Path):
    from src.evaluation.pre_dataset_capability import write_pre_dataset_capability_outputs

    rules_path = tmp_path / "rules.yaml"
    output_dir = tmp_path / "reports"
    _write_rules(
        rules_path,
        orthography={
            "model_required_fixture": _entry(
                status="model_required",
                rule_ids=["final_punctuation_default"],
                requires=["model"],
                eligible_now=False,
            ),
            "metadata_group": _entry(entry_type="group", status="metadata_only", executable=False, requires=[]),
        },
    )

    outputs = write_pre_dataset_capability_outputs(
        rules_config_path=rules_path,
        output_dir=output_dir,
        matrix_reports_dir=tmp_path / "missing",
        update_rules_yaml=True,
    )

    assert REQUIRED_PRE_DATASET_REPORTS <= {Path(path).name for path in outputs.values()}
    assert REQUIRED_PRE_DATASET_REPORTS <= {path.name for path in output_dir.iterdir()}

    updated = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    dataset = updated["orthography"]["model_required_fixture"]["dataset"]
    assert dataset["production_ready_now"] is False
    assert dataset["training_eligible_now"] is True
    assert dataset["training_eligibility_decision"] in {
        "INCLUDE_NOW",
        "INCLUDE_AFTER_VALIDATOR",
        "INCLUDE_AFTER_THRESHOLD_CALIBRATION",
        "INCLUDE_AFTER_TRAINING",
    }
    assert updated["orthography"]["metadata_group"]["dataset"]["training_eligibility_decision"] == "BLOCK_METADATA_ONLY"


def test_pre_dataset_cli_help_loads_project_imports():
    completed = subprocess.run(
        [sys.executable, "scripts/run_pre_dataset_capability.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert "pre-dataset capability audit" in completed.stdout


def test_current_project_pre_dataset_reports_are_consistent_after_generation():
    rules_path = Path("configs/rules.yaml")
    report_dir = Path("reports/pre_dataset_capability")

    assert rules_path.exists()
    assert REQUIRED_PRE_DATASET_REPORTS <= {path.name for path in report_dir.iterdir()}

    config = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    entries = [entry for section in ("orthography", "punctuation") for entry in config[section].values()]
    strict_count = sum(1 for entry in entries if entry["dataset"].get("eligible_now"))
    training_count = sum(1 for entry in entries if entry["dataset"].get("training_eligible_now"))
    active_rule_ids = {
        rule_id
        for entry in entries
        if entry["dataset"].get("training_eligible_now")
        for rule_id in entry["implementation"].get("rule_ids", [])
    }

    assert len(entries) == 409
    assert training_count >= strict_count or "explicit audited justification" in (
        report_dir / "current_capability_summary.md"
    ).read_text(encoding="utf-8")
    assert len(active_rule_ids) > strict_count

    eligible = pd.read_csv(report_dir / "dataset_eligible_rules.csv")
    blocked = pd.read_csv(report_dir / "dataset_blocked_rules.csv")
    probe = pd.read_csv(report_dir / "candidate_path_probe.csv")
    assert not eligible.empty
    assert not blocked.empty
    assert not probe.empty
    assert {"rule_id", "training_eligibility_decision", "current_candidate_recall"} <= set(eligible.columns)
    assert {"rule_id", "generated_candidate_count", "matching_candidate_found", "candidate_recall"} <= set(probe.columns)
