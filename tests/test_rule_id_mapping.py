from __future__ import annotations

import csv
from pathlib import Path

import yaml

from src.rules.registry import all_rules
from src.rules.rule_ids import normalize_rule_id


RULES_PATH = Path("configs/rules.yaml")
MAPPING_PATH = Path("reports/rules_taxonomy/current_rule_id_mapping.csv")

REQUIRED_COLUMNS = {
    "rule_id",
    "canonical_rule_id",
    "source_module",
    "mapped_matrix_key",
    "mapped_orfogrammka_id",
    "mapped_title",
    "mapping_confidence",
    "mapping_reason",
    "action",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low", "unmapped"}
ALLOWED_ACTIONS = {"mapped", "needs_manual_review", "missing_from_taxonomy", "alias", "deprecated"}


def _read_mapping() -> list[dict[str, str]]:
    with MAPPING_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _matrix_keys() -> set[str]:
    config = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    return set(config["orthography"]) | set(config["punctuation"])


def test_current_rule_id_mapping_report_exists_and_has_required_columns():
    assert MAPPING_PATH.exists()
    rows = _read_mapping()

    assert rows
    assert REQUIRED_COLUMNS <= set(rows[0])


def test_all_registry_rule_ids_are_mapped_or_reported():
    rows = _read_mapping()
    canonical_ids = {normalize_rule_id(row["canonical_rule_id"]) for row in rows}
    registry_ids = {rule.spec.id for rule in all_rules()}

    assert registry_ids <= canonical_ids


def test_mapping_actions_and_confidence_values_are_allowed():
    for row in _read_mapping():
        assert row["mapping_confidence"] in ALLOWED_CONFIDENCE
        assert row["action"] in ALLOWED_ACTIONS


def test_mapped_rule_ids_point_to_existing_matrix_keys():
    keys = _matrix_keys()

    for row in _read_mapping():
        if row["action"] not in {"mapped", "needs_manual_review"}:
            continue
        assert row["mapped_matrix_key"] in keys


def test_unmapped_rows_do_not_claim_matrix_target():
    for row in _read_mapping():
        if row["mapping_confidence"] == "unmapped":
            assert not row["mapped_matrix_key"]
            assert row["action"] in {"missing_from_taxonomy", "deprecated"}
