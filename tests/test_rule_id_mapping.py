from __future__ import annotations

from pathlib import Path

from src.grammar_gen.rules.registry import default_rule_registry
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.rule_ids import normalize_rule_id
from src.schema.labels import RULE_LABEL_TO_ID


RULES_PATH = Path("configs/rules.yaml")


def _coverage_rule_ids() -> set[str]:
    data = load_rules_coverage(RULES_PATH)
    return {
        normalize_rule_id(rule_id)
        for _domain, _group, entry in iter_coverage_entries(data)
        for rule_id in entry.get("rules", [])
    }


def test_rules_yaml_replaces_generated_rule_id_mapping_report() -> None:
    data = load_rules_coverage(RULES_PATH)

    assert isinstance(data["orthography"], dict)
    assert isinstance(data["punctuation"], dict)
    assert _coverage_rule_ids()


def test_direct_generation_rule_ids_are_known_direct_labels() -> None:
    direct_rule_ids = {rule.info.rule_id for rule in default_rule_registry().all_rules()}

    assert direct_rule_ids
    assert direct_rule_ids <= set(RULE_LABEL_TO_ID)


def test_configured_rule_ids_are_normalized_and_non_empty() -> None:
    rule_ids = _coverage_rule_ids()

    assert "" not in rule_ids
    assert all(rule_id == normalize_rule_id(rule_id) for rule_id in rule_ids)


def test_rules_matrix_contains_current_direct_runtime_anchors() -> None:
    rule_ids = _coverage_rule_ids()

    assert "ne_verb" in rule_ids
    assert "hyphen_particles" in rule_ids
    assert "comma_subordinate" in rule_ids
