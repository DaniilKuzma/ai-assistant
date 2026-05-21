from pathlib import Path

import pandas as pd

from src.evaluation.matrix_inventory import (
    INVENTORY_COLUMNS,
    build_rule_id_alias_audit,
    build_rule_matrix_inventory,
    write_rule_matrix_inventory_outputs,
)


def test_rule_matrix_inventory_covers_every_yaml_group_and_decides(tmp_path: Path):
    inventory = build_rule_matrix_inventory(
        rules_config_path="configs/rules.yaml",
        reports_dir=tmp_path / "missing_reports",
        dataset_manifest_path=tmp_path / "missing_manifest.json",
        current_dataset_path=tmp_path / "missing_dataset.csv.gz",
    )

    assert list(inventory.columns) == INVENTORY_COLUMNS
    assert len(inventory) == 72
    assert set(inventory["section"]) == {"orthography", "punctuation"}
    assert inventory["decision"].ne("").all()

    implemented = inventory[inventory["status_from_yaml"] == "implemented"]
    assert not implemented.empty
    assert implemented["registry_rule_exists"].all()

    planned = inventory[inventory["status_from_yaml"] == "planned"]
    assert not planned.empty
    assert set(planned["decision"]) <= {"BACKLOG_IMPLEMENTATION", "BACKLOG_DATA"}


def test_rule_matrix_inventory_outputs_include_summary(tmp_path: Path):
    result = write_rule_matrix_inventory_outputs(
        output_dir=tmp_path,
        rules_config_path="configs/rules.yaml",
        reports_dir=tmp_path / "missing_reports",
        dataset_manifest_path=tmp_path / "missing_manifest.json",
        current_dataset_path=tmp_path / "missing_dataset.csv.gz",
    )

    inventory_path = Path(result["inventory_path"])
    summary_path = Path(result["summary_path"])
    alias_path = Path(result["alias_audit_path"])

    assert inventory_path.exists()
    assert summary_path.exists()
    assert alias_path.exists()

    inventory = pd.read_csv(inventory_path)
    assert len(inventory) == 72
    text = summary_path.read_text(encoding="utf-8")
    assert "total matrix groups" in text
    assert "groups eligible for next dataset cycle" in text


def test_rule_id_alias_audit_marks_legacy_alias_and_report_only_unknown():
    audit = build_rule_id_alias_audit(
        matrix_rule_ids=["frequent_error_exact"],
        registry_rule_ids=["frequent_error_exact"],
        source_rule_ids={
            "reports": ["frequent_errors", "unknown"],
            "dataset": ["synthetic_only_id"],
        },
    )

    by_raw = {(row["source"], row["raw_rule_id"]): row for row in audit.to_dict("records")}
    assert by_raw[("reports", "frequent_errors")]["canonical_rule_id"] == "frequent_error_exact"
    assert by_raw[("reports", "frequent_errors")]["action"] == "alias"
    assert by_raw[("reports", "unknown")]["action"] == "unknown"
    assert by_raw[("dataset", "synthetic_only_id")]["action"] == "missing_in_matrix"
