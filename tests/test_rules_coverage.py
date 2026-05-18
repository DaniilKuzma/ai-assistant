from collections import Counter
from pathlib import Path

import pytest
import yaml


RULES_PATH = Path("configs/rules.yaml")
REQUIRED_ENTRY_FIELDS = {
    "orfogrammka_id",
    "title",
    "parent_group",
    "status",
    "requires",
    "rules",
    "notes",
}
ORTHOGRAPHY_PARENT_GROUPS = {
    "hissing_spelling",
    "hard_soft_sign",
    "prefix_spelling",
    "roots",
    "suffixes",
    "endings",
    "ne_ni",
    "n_nn",
    "checked_unchecked_vowels",
    "checked_unchecked_consonants",
    "double_consonants",
    "dictionary_words",
    "borrowed_words",
    "hyphen_slitno_razdelno",
    "capitalization",
    "abbreviations",
    "typos",
}
PUNCTUATION_PARENT_GROUPS = {
    "final_punctuation",
    "punctuation_inside_sentence",
    "comma_subordinate",
    "comma_conjunction",
    "homogeneous_members",
    "detached_members",
    "introductory_words",
    "addresses",
    "comparative_turnovers",
    "subject_predicate_dash",
    "direct_speech",
    "quotes_brackets",
    "colon_dash_semicolon",
    "punctuation_combinations",
    "neural_punctuation",
}


def _coverage_entry(
    *,
    title: str,
    status: str,
    requires: list[str],
    rules: list[str],
    notes: str,
    orfogrammka_id: str = "test.fixture",
    parent_group: str = "dictionary_words",
) -> dict[str, object]:
    return {
        "orfogrammka_id": orfogrammka_id,
        "title": title,
        "parent_group": parent_group,
        "status": status,
        "requires": requires,
        "rules": rules,
        "notes": notes,
    }


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


def test_all_statuses_are_allowed():
    from src.rules.coverage_matrix import ALLOWED_STATUSES, iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)

    for domain, group, entry in iter_coverage_entries(data):
        assert entry["status"] in ALLOWED_STATUSES, f"{domain}.{group} uses an unknown status"


def test_all_coverage_entries_have_required_taxonomy_fields():
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)

    for domain, group, entry in iter_coverage_entries(data):
        assert REQUIRED_ENTRY_FIELDS <= set(entry), f"{domain}.{group} is missing taxonomy fields"
        assert entry["orfogrammka_id"].strip(), f"{domain}.{group}.orfogrammka_id must be non-empty"
        assert entry["parent_group"].strip(), f"{domain}.{group}.parent_group must be non-empty"


def test_all_required_parent_groups_are_present():
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage

    data = load_rules_coverage(RULES_PATH)
    parents_by_domain = {"orthography": set(), "punctuation": set()}
    for domain, _group, entry in iter_coverage_entries(data):
        parents_by_domain[domain].add(entry["parent_group"])

    assert ORTHOGRAPHY_PARENT_GROUPS <= parents_by_domain["orthography"]
    assert PUNCTUATION_PARENT_GROUPS <= parents_by_domain["punctuation"]


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
                    "future_dictionary_rule": _coverage_entry(
                        title="Future dictionary rule",
                        status="planned",
                        requires=["dictionary"],
                        rules=[],
                        notes="Metadata-only entry.",
                    )
                },
                "punctuation": {
                    "future_syntax_rule": _coverage_entry(
                        title="Future syntax rule",
                        status="syntax_required",
                        requires=["syntax"],
                        rules=[],
                        notes="Metadata-only entry.",
                        parent_group="punctuation_inside_sentence",
                    ),
                    "future_model_rule": _coverage_entry(
                        title="Future model rule",
                        status="model_required",
                        requires=["model"],
                        rules=[],
                        notes="Metadata-only entry.",
                        parent_group="neural_punctuation",
                    ),
                    "future_dictionary_model_rule": _coverage_entry(
                        title="Future dictionary model rule",
                        status="dictionary_model_required",
                        requires=["dictionary", "model"],
                        rules=[],
                        notes="Metadata-only entry.",
                        parent_group="punctuation_combinations",
                    ),
                    "future_ner_rule": _coverage_entry(
                        title="Future NER rule",
                        status="ner_required",
                        requires=["ner"],
                        rules=[],
                        notes="Metadata-only entry.",
                        parent_group="addresses",
                    )
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    validate_rules_coverage(path)


def test_metadata_only_statuses_may_list_non_registry_rule_ids(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "future_dictionary_rule": _coverage_entry(
                        title="Future dictionary rule",
                        status="dictionary_model_required",
                        requires=["dictionary", "model"],
                        rules=["future_dictionary_metadata_id"],
                        notes="Metadata-only entry.",
                    )
                },
                "punctuation": {
                    "future_syntax_rule": _coverage_entry(
                        title="Future syntax rule",
                        status="syntax_required",
                        requires=["syntax"],
                        rules=["future_punctuation_metadata_id"],
                        notes="Metadata-only entry.",
                        parent_group="punctuation_inside_sentence",
                    )
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
                    "fake_implemented_rule": _coverage_entry(
                        title="Fake implemented rule",
                        status="implemented",
                        requires=[],
                        rules=["not_a_real_rule_id"],
                        notes="This must fail.",
                    )
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


def test_candidate_only_and_partial_groups_require_real_rule_ids(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "fake_candidate_only_rule": _coverage_entry(
                        title="Fake candidate-only rule",
                        status="candidate_only",
                        requires=["model"],
                        rules=["not_a_real_candidate_rule_id"],
                        notes="This must fail.",
                    ),
                    "fake_partial_rule": _coverage_entry(
                        title="Fake partial rule",
                        status="partial",
                        requires=["dictionary"],
                        rules=["not_a_real_partial_rule_id"],
                        notes="This must fail.",
                    ),
                },
                "punctuation": {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not_a_real_candidate_rule_id"):
        validate_rules_coverage(path)


def test_unknown_dependency_is_rejected(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "bad_dependency": _coverage_entry(
                        title="Bad dependency",
                        status="planned",
                        requires=["regex_dictionary"],
                        rules=[],
                        notes="This dependency is not part of the roadmap contract.",
                    )
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


def test_unknown_status_is_rejected(tmp_path):
    from src.rules.coverage_matrix import validate_rules_coverage

    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orthography": {
                    "bad_status": _coverage_entry(
                        title="Bad status",
                        status="implemented_later",
                        requires=[],
                        rules=[],
                        notes="This status is not part of the coverage contract.",
                    )
                },
                "punctuation": {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="implemented_later"):
        validate_rules_coverage(path)


def test_all_listed_rule_ids_are_unique_except_multi_taxonomy_ids_and_registry_ids_are_unique():
    from src.rules.coverage_matrix import MULTI_TAXONOMY_RULE_IDS, iter_rule_ids, load_rules_coverage
    from src.rules.registry import all_rules

    listed_rule_ids = list(iter_rule_ids(load_rules_coverage(RULES_PATH)))
    registry_rule_ids = [rule.spec.id for rule in all_rules()]
    duplicated_listed_ids = {rule_id for rule_id, count in Counter(listed_rule_ids).items() if count > 1}

    assert duplicated_listed_ids <= MULTI_TAXONOMY_RULE_IDS
    assert len(registry_rule_ids) == len(set(registry_rule_ids))


def test_rule_expansion_plan_exists_and_is_not_empty():
    path = Path("docs/rule_expansion_plan.md")

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()
