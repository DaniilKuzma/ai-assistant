from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


RULES_COVERAGE_PATH = Path(__file__).resolve().parents[2] / "configs" / "rules.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_STATUSES = frozenset(
    {
        "implemented",
        "partial",
        "planned",
        "deterministic",
        "candidate_only",
        "model_required",
        "syntax_required",
        "dictionary_model_required",
        "ner_required",
    }
)
ALLOWED_DEPENDENCIES = frozenset(
    {
        "morphology",
        "syntax",
        "dictionary",
        "ner",
        "model",
        "validator",
        "frequency_lexicon",
    }
)
EXECUTABLE_STATUSES = frozenset({"implemented", "deterministic"})
METADATA_ONLY_STATUSES = frozenset(
    {
        "planned",
        "model_required",
        "syntax_required",
        "dictionary_model_required",
        "ner_required",
    }
)
REQUIRED_COVERAGE_TAGS = frozenset(
    {
        "orthography",
        "punctuation",
        "dictionary_errors",
        "morphological_rules",
        "syntactic_rules",
        "context_pairs",
        "capitalization",
        "hyphen_solid_separate",
        "final_punctuation",
        "complex_punctuation",
        "direct_speech",
        "introductory_constructions",
        "detached_members",
        "homogeneous_members",
        "punctuation_combinations",
    }
)
REQUIRED_SECTIONS = ("orthography", "punctuation")
REQUIRED_GROUP_FIELDS = frozenset({"title", "status", "requires", "rules", "notes"})


def load_rules_coverage(path: str | Path = RULES_COVERAGE_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("rules coverage matrix must be a YAML mapping")
    return data


def iter_coverage_entries(data: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for domain in REQUIRED_SECTIONS:
        groups = data.get(domain, {})
        if not isinstance(groups, dict):
            continue
        for group, entry in groups.items():
            if isinstance(entry, dict):
                yield domain, str(group), entry


def iter_rule_ids(data: dict[str, Any]) -> Iterable[str]:
    for _domain, _group, entry in iter_coverage_entries(data):
        for rule_id in entry.get("rules", []):
            yield str(rule_id)


def validate_rules_coverage(
    path: str | Path = RULES_COVERAGE_PATH,
    *,
    registry_rule_ids: set[str] | None = None,
) -> dict[str, Any]:
    data = load_rules_coverage(path)
    _validate_sections(data)

    if registry_rule_ids is None:
        from src.rules.registry import all_rules

        registry_rule_ids = {rule.spec.id for rule in all_rules()}

    seen_rule_ids: set[str] = set()
    for domain, group, entry in iter_coverage_entries(data):
        _validate_group(domain, group, entry)
        status = entry["status"]
        rules = entry["rules"]
        if status in EXECUTABLE_STATUSES and not rules:
            raise ValueError(f"{domain}.{group} is implemented but has no rule_id")
        for rule_id in rules:
            if rule_id in seen_rule_ids:
                raise ValueError(f"Duplicate rule_id in coverage matrix: {rule_id}")
            if rule_id not in registry_rule_ids:
                raise ValueError(f"{domain}.{group} references unknown rule_id: {rule_id}")
            seen_rule_ids.add(rule_id)
    return data


def validate_project_rules_coverage(path: str | Path = RULES_COVERAGE_PATH) -> dict[str, Any]:
    data = validate_rules_coverage(path)
    _validate_required_tags(data)
    _validate_implemented_tests(data, Path(path).resolve().parent.parent)
    return data


def _validate_sections(data: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in data:
            raise ValueError(f"Missing rules coverage section: {section}")
        if not isinstance(data[section], dict):
            raise ValueError(f"Rules coverage section must be a mapping: {section}")


def _validate_group(domain: str, group: str, entry: dict[str, Any]) -> None:
    missing = REQUIRED_GROUP_FIELDS - set(entry)
    if missing:
        raise ValueError(f"{domain}.{group} is missing required fields: {sorted(missing)}")
    if not isinstance(entry["title"], str) or not entry["title"].strip():
        raise ValueError(f"{domain}.{group}.title must be a non-empty string")
    if entry["status"] not in ALLOWED_STATUSES:
        raise ValueError(f"{domain}.{group}.status is not allowed: {entry['status']}")
    if not isinstance(entry["requires"], list) or not all(isinstance(item, str) for item in entry["requires"]):
        raise ValueError(f"{domain}.{group}.requires must be a list of strings")
    unknown_dependencies = sorted(set(entry["requires"]) - ALLOWED_DEPENDENCIES)
    if unknown_dependencies:
        raise ValueError(f"{domain}.{group}.requires contains unknown dependencies: {unknown_dependencies}")
    if not isinstance(entry["rules"], list) or not all(isinstance(item, str) for item in entry["rules"]):
        raise ValueError(f"{domain}.{group}.rules must be a list of rule_id strings")
    if not isinstance(entry["notes"], str):
        raise ValueError(f"{domain}.{group}.notes must be a string")
    if "tags" in entry and (
        not isinstance(entry["tags"], list) or not all(isinstance(item, str) and item for item in entry["tags"])
    ):
        raise ValueError(f"{domain}.{group}.tags must be a list of non-empty strings")
    if "tests" in entry and (
        not isinstance(entry["tests"], list) or not all(isinstance(item, str) and item for item in entry["tests"])
    ):
        raise ValueError(f"{domain}.{group}.tests must be a list of pytest marker strings")


def _validate_required_tags(data: dict[str, Any]) -> None:
    tags = {
        tag
        for _domain, _group, entry in iter_coverage_entries(data)
        for tag in entry.get("tags", [])
    }
    missing = sorted(REQUIRED_COVERAGE_TAGS - tags)
    if missing:
        raise ValueError(f"Rules coverage matrix is missing required tags: {missing}")


def _validate_implemented_tests(data: dict[str, Any], project_root: Path = PROJECT_ROOT) -> None:
    for domain, group, entry in iter_coverage_entries(data):
        if entry["status"] not in EXECUTABLE_STATUSES:
            continue
        tests = entry.get("tests", [])
        if not tests:
            raise ValueError(f"{domain}.{group} is implemented but has no tests markers")
        for marker in tests:
            if not _test_marker_exists(marker, project_root):
                raise ValueError(f"{domain}.{group} references missing test marker: {marker}")


def _test_marker_exists(marker: str, project_root: Path) -> bool:
    path_text, _separator, selector = marker.partition("::")
    test_path = project_root / path_text
    if not test_path.exists():
        return False
    if not selector:
        return True
    selector_name = selector.split("::")[-1].split("[", 1)[0]
    text = test_path.read_text(encoding="utf-8")
    return f"def {selector_name}(" in text or f"class {selector_name}" in text
