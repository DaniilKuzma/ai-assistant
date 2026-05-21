from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml


RULES_PATH = Path("configs/rules.yaml")
STRICT_STATUSES = {
    "implemented",
    "partial",
    "candidate_only",
    "model_required",
    "syntax_required",
    "dictionary_model_required",
    "ner_required",
    "planned",
    "metadata_only",
    "disabled",
}


def _v3_entry(
    *,
    section: str = "orthography",
    title: str = "Future dictionary rule",
    status: str = "planned",
    requires: list[str] | None = None,
    rule_ids: list[str] | None = None,
    executable: bool = False,
) -> dict[str, object]:
    return {
        "source_section": section,
        "orfogrammka_id": "test.fixture",
        "title": title,
        "normalized_title": title.lower(),
        "entry_type": "leaf_rule",
        "parent_key": f"{section}_group_fixture",
        "parent_path": ["Fixture"],
        "depth": 2,
        "order": 1,
        "source_url": "https://orfogrammka.ru/fixture/",
        "implementation": {
            "status": status,
            "executable": executable,
            "rule_ids": rule_ids or [],
            "aliases": [],
            "requires": requires or [],
            "notes": "Fixture entry.",
        },
        "dataset": {
            "eligible_now": False,
            "reason": "planned",
            "last_known_candidate_recall": None,
            "last_known_eval_count": None,
            "production_ready_now": False,
            "training_eligible_now": False,
            "training_eligibility_decision": "BLOCK_PLANNED",
            "training_eligibility_reason": "planned",
            "current_candidate_path": False,
            "current_synthetic_support": False,
            "current_hard_negative_support": False,
            "current_validator_support": False,
            "current_candidate_recall": None,
            "current_gap_coverage": None,
            "risk_level": "high",
            "needs_before_training": ["none"],
        },
    }


def _write_fixture(path: Path, orthography: dict, punctuation: dict) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "source": {"taxonomy_provider": "orfogrammka"},
                "orthography": orthography,
                "punctuation": punctuation,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_rules_yaml_exists():
    assert RULES_PATH.exists()


def test_rules_yaml_is_valid_and_has_required_sections():
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    assert data["schema_version"] >= 3
    assert isinstance(data["orthography"], dict)
    assert isinstance(data["punctuation"], dict)


def test_rules_coverage_matrix_validates_against_rule_registry():
    from src.rules.coverage_matrix import validate_rules_coverage

    validate_rules_coverage(RULES_PATH)


def test_project_rules_coverage_checker_passes():
    from src.rules.coverage_matrix import validate_project_rules_coverage

    validate_project_rules_coverage(RULES_PATH)


def test_training_eligibility_fields_are_present_and_block_no_candidate_entries():
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)
    for domain, group, entry in iter_coverage_entries(data):
        dataset = entry["dataset"]
        assert "production_ready_now" in dataset, f"{domain}.{group} missing production_ready_now"
        assert "training_eligible_now" in dataset, f"{domain}.{group} missing training_eligible_now"
        if not entry["rules"]:
            assert dataset["training_eligible_now"] is False, f"{domain}.{group} lacks rule_ids"


def test_all_statuses_are_allowed():
    from src.rules.coverage_matrix import ALLOWED_STATUSES, iter_coverage_entries, load_rules_coverage

    assert ALLOWED_STATUSES == STRICT_STATUSES
    data = load_rules_coverage(RULES_PATH)

    for domain, group, entry in iter_coverage_entries(data):
        assert entry["status"] in ALLOWED_STATUSES, f"{domain}.{group} uses an unknown status"


def test_iter_coverage_entries_keeps_legacy_flat_fields_for_callers():
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)
    domain, _group, entry = next(iter(iter_coverage_entries(data)))

    assert domain in {"orthography", "punctuation"}
    assert {"orfogrammka_id", "title", "parent_group", "status", "requires", "rules", "notes"} <= set(entry)


def test_implemented_rule_ids_exist_in_registry():
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
    from src.rules.registry import rule_by_id

    data = load_rules_coverage(RULES_PATH)
    implemented_entries = [
        (domain, group, entry)
        for domain, group, entry in iter_coverage_entries(data)
        if entry["status"] == "implemented"
    ]

    assert implemented_entries
    for domain, group, entry in implemented_entries:
        assert entry["rules"], f"{domain}.{group} must list at least one rule_id"
        for rule_id in entry["rules"]:
            assert rule_by_id(rule_id) is not None


def test_implemented_groups_have_test_markers():
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)
    implemented_entries = [
        (domain, group, entry)
        for domain, group, entry in iter_coverage_entries(data)
        if entry["status"] == "implemented"
    ]

    assert implemented_entries
    for domain, group, entry in implemented_entries:
        assert entry.get("tests"), f"{domain}.{group} must list pytest coverage markers"


def test_planned_and_metadata_statuses_do_not_require_executable_rules(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    _write_fixture(
        path,
        {"future_dictionary_rule": _v3_entry(status="planned", requires=["dictionary"], rule_ids=[])},
        {
            "future_syntax_rule": _v3_entry(section="punctuation", status="syntax_required", requires=["syntax"], rule_ids=[]),
            "future_metadata_group": _v3_entry(section="punctuation", status="metadata_only", requires=[], rule_ids=[]),
        },
    )

    validate_rules_coverage(path)


def test_executable_entries_require_real_rule_ids(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    _write_fixture(path, {"fake_implemented_rule": _v3_entry(status="implemented", rule_ids=["not_a_real_rule_id"], executable=True)}, {})

    with pytest.raises(ValueError, match="not_a_real_rule_id"):
        validate_rules_coverage(path)


def test_candidate_only_and_partial_entries_require_real_rule_ids(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    _write_fixture(
        path,
        {
            "fake_candidate_only_rule": _v3_entry(status="candidate_only", requires=["model"], rule_ids=["not_a_real_candidate_rule_id"], executable=True),
            "fake_partial_rule": _v3_entry(status="partial", requires=["dictionary"], rule_ids=["not_a_real_partial_rule_id"], executable=True),
        },
        {},
    )

    with pytest.raises(ValueError, match="not_a_real_candidate_rule_id"):
        validate_rules_coverage(path)


def test_unknown_dependency_is_rejected(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    _write_fixture(path, {"bad_dependency": _v3_entry(status="planned", requires=["regex_dictionary"])}, {})

    with pytest.raises(ValueError, match="regex_dictionary"):
        validate_rules_coverage(path)


def test_unknown_status_is_rejected(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    _write_fixture(path, {"bad_status": _v3_entry(status="implemented_later")}, {})

    with pytest.raises(ValueError, match="implemented_later"):
        validate_rules_coverage(path)


def test_all_listed_rule_ids_are_unique_and_registry_ids_are_unique():
    from src.rules.coverage_matrix import iter_rule_ids, load_rules_coverage
    from src.rules.registry import all_rules

    listed_rule_ids = list(iter_rule_ids(load_rules_coverage(RULES_PATH)))
    registry_rule_ids = [rule.spec.id for rule in all_rules()]
    duplicated_listed_ids = {rule_id for rule_id, count in Counter(listed_rule_ids).items() if count > 1}

    assert not duplicated_listed_ids
    assert len(registry_rule_ids) == len(set(registry_rule_ids))


def test_rule_expansion_plan_exists_and_is_not_empty():
    path = Path("docs/rule_expansion_plan.md")

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()
