from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.rules.orthography import orthography_rules as _orthography_rules
from src.rules.punctuation import punctuation_rules as _punctuation_rules
from src.rules.synthetic import DEFAULT_ORTHOGRAPHY_BALANCE, DEFAULT_PUNCTUATION_BALANCE


def all_rules() -> tuple[Any, ...]:
    return (*orthography_rules(), *punctuation_rules())


def orthography_rules() -> tuple[Any, ...]:
    return _orthography_rules()


def punctuation_rules() -> tuple[Any, ...]:
    return _punctuation_rules()


def synthetic_rules() -> tuple[Any, ...]:
    groups = set(DEFAULT_ORTHOGRAPHY_BALANCE) | set(DEFAULT_PUNCTUATION_BALANCE) | {"frequent_spelling", "split_join", "hyphen"}
    return tuple(rule for rule in all_rules() if rule.spec.group in groups or rule.spec.id in groups)


@lru_cache(maxsize=1)
def _rules_by_id() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rule in all_rules():
        if rule.spec.id in result:
            raise ValueError(f"Duplicate rule id: {rule.spec.id}")
        result[rule.spec.id] = rule
    return result


def rule_by_id(rule_id: str) -> Any | None:
    return _rules_by_id().get(rule_id)


def coverage_status() -> dict[str, dict[str, str]]:
    from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage

    coverage: dict[str, dict[str, str]] = {}
    for domain, group, entry in iter_coverage_entries(load_rules_coverage()):
        requires = ", ".join(entry["requires"]) if entry["requires"] else "none"
        coverage[group] = {
            "domain": domain,
            "status": entry["status"],
            "requires": requires,
            "description": entry["notes"],
        }
    return coverage
