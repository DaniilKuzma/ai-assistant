from __future__ import annotations

import pytest

import test_rule_layers_coverage as rule_coverage


def test_enabled_rule_layer_groups_load_specs_with_capabilities_metadata_and_schema_labels() -> None:
    rule_coverage.test_enabled_rule_layer_groups_load_specs_with_capabilities_metadata_and_schema_labels()


@pytest.mark.parametrize("rule_id", sorted(rule_coverage.GUARD_ONLY_RULE_IDS))
def test_guard_only_layer_rules_generate_identity_guards_and_reject_positive(rule_id: str) -> None:
    rule_coverage.test_guard_only_layer_rules_generate_identity_guards_and_reject_positive(rule_id)
