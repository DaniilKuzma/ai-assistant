from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


RULES_PATH = Path("configs/rules.yaml")
REPORT_DIR = Path("reports/syntax_module")
INVENTORY_CSV = REPORT_DIR / "syntax_required_inventory.csv"
FAMILY_SUMMARY_CSV = REPORT_DIR / "syntax_required_family_summary.csv"
IMPLEMENTATION_PLAN = REPORT_DIR / "syntax_module_implementation_plan.md"


def _syntax_required_entries() -> dict[str, dict]:
    config = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    entries: dict[str, dict] = {}
    for section in ("orthography", "punctuation"):
        for matrix_key, entry in config[section].items():
            implementation = entry.get("implementation") or {}
            if implementation.get("status") == "syntax_required":
                entries[matrix_key] = entry
    return entries


def test_syntax_required_inventory_report_exists_and_covers_rules_yaml():
    assert INVENTORY_CSV.exists()

    inventory = pd.read_csv(INVENTORY_CSV)
    entries = _syntax_required_entries()

    assert set(inventory["matrix_key"]) == set(entries)
    assert len(inventory) == len(entries)
    assert not inventory["matrix_key"].duplicated().any()


def test_syntax_required_inventory_rows_have_required_classification_fields():
    inventory = pd.read_csv(INVENTORY_CSV)

    assert inventory["syntax_family"].fillna("").str.strip().ne("").all()
    assert inventory["recommended_action"].fillna("").str.strip().ne("").all()
    assert inventory["parent_path"].fillna("").str.strip().ne("").all()
    assert inventory["title"].fillna("").str.strip().ne("").all()


def test_syntax_required_summary_and_implementation_plan_exist():
    assert FAMILY_SUMMARY_CSV.exists()
    assert IMPLEMENTATION_PLAN.exists()

    summary = pd.read_csv(FAMILY_SUMMARY_CSV)
    plan_text = IMPLEMENTATION_PLAN.read_text(encoding="utf-8")

    assert len(summary) == 25
    assert {"syntax_family", "total_entries", "priority"} <= set(summary.columns)
    assert "SYNTAX_REQUIRED_AUDIT_COMPLETE" in plan_text
