from pathlib import Path

import pytest
import yaml


RULES_PATH = Path("configs/rules.yaml")


def test_rules_yaml_exists():
    assert RULES_PATH.exists()


def test_rules_yaml_is_valid_and_has_required_sections():
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    assert isinstance(data["orthography"], dict)
    assert isinstance(data["punctuation"], dict)


def test_rules_coverage_matrix_validates_against_rule_registry():
    from src.rules.coverage_matrix import validate_rules_coverage

    validate_rules_coverage(RULES_PATH)


def test_project_rules_coverage_checker_passes():
    from src.rules.coverage_matrix import validate_project_rules_coverage

    validate_project_rules_coverage(RULES_PATH)


def test_all_required_coverage_tags_are_present():
    from src.rules.coverage_matrix import REQUIRED_COVERAGE_TAGS, iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)
    tags = {
        tag
        for _domain, _group, entry in iter_coverage_entries(data)
        for tag in entry.get("tags", [])
    }

    assert REQUIRED_COVERAGE_TAGS <= tags


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
        if entry["status"] in {"implemented", "deterministic"}
    ]

    assert implemented_entries
    for domain, group, entry in implemented_entries:
        assert entry.get("tests"), f"{domain}.{group} must list pytest coverage markers"


def test_planned_groups_do_not_require_executable_rules(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "future_dictionary_rule": {
                        "title": "Future dictionary rule",
                        "status": "planned",
                        "requires": ["dictionary"],
                        "rules": [],
                        "notes": "Metadata-only entry.",
                    }
                },
                "punctuation": {
                    "future_syntax_rule": {
                        "title": "Future syntax rule",
                        "status": "syntax_required",
                        "requires": ["syntax"],
                        "rules": [],
                        "notes": "Metadata-only entry.",
                    },
                    "future_model_rule": {
                        "title": "Future model rule",
                        "status": "model_required",
                        "requires": ["model"],
                        "rules": [],
                        "notes": "Metadata-only entry.",
                    },
                    "future_dictionary_model_rule": {
                        "title": "Future dictionary model rule",
                        "status": "dictionary_model_required",
                        "requires": ["dictionary", "model"],
                        "rules": [],
                        "notes": "Metadata-only entry.",
                    },
                    "future_ner_rule": {
                        "title": "Future NER rule",
                        "status": "ner_required",
                        "requires": ["ner"],
                        "rules": [],
                        "notes": "Metadata-only entry.",
                    }
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    validate_rules_coverage(path)


def test_implemented_groups_require_real_rule_ids(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "fake_implemented_rule": {
                        "title": "Fake implemented rule",
                        "status": "implemented",
                        "requires": [],
                        "rules": ["not_a_real_rule_id"],
                        "notes": "This must fail.",
                    }
                },
                "punctuation": {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not_a_real_rule_id"):
        validate_rules_coverage(path)


def test_unknown_dependency_is_rejected(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "bad_dependency": {
                        "title": "Bad dependency",
                        "status": "planned",
                        "requires": ["regex_dictionary"],
                        "rules": [],
                        "notes": "This dependency is not part of the roadmap contract.",
                    }
                },
                "punctuation": {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="regex_dictionary"):
        validate_rules_coverage(path)


def test_all_listed_rule_ids_are_unique_and_registry_ids_are_unique():
    from src.rules.coverage_matrix import iter_rule_ids, load_rules_coverage
    from src.rules.registry import all_rules

    listed_rule_ids = list(iter_rule_ids(load_rules_coverage(RULES_PATH)))
    registry_rule_ids = [rule.spec.id for rule in all_rules()]

    assert len(listed_rule_ids) == len(set(listed_rule_ids))
    assert len(registry_rule_ids) == len(set(registry_rule_ids))


def test_rule_expansion_plan_exists_and_is_not_empty():
    path = Path("docs/rule_expansion_plan.md")

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()
