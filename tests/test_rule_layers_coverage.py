from __future__ import annotations

from src.rule_layers.base import LayerDirectCase, LayerRuleSpec
from src.rule_layers.coverage import collect_layer_coverage, validate_layer_coverage


def test_collect_layer_coverage_groups_by_layer_rule_and_sub_rule() -> None:
    specs = (
        LayerRuleSpec(
            layer="compound",
            rule_id="ne_verb",
            family="compound_spelling",
            cases=(
                _case("ne_verb", "compound_spelling", "merge", "positive"),
                _case("ne_verb", "compound_spelling", "merge", "hard_negative"),
            ),
        ),
    )

    coverage = collect_layer_coverage(specs)

    assert coverage == {"compound": {"ne_verb": {"merge": {"positive": 1, "hard_negative": 1}}}}


def test_validate_layer_coverage_requires_positive_and_non_positive_case() -> None:
    specs = (
        LayerRuleSpec(
            layer="compound",
            rule_id="ne_verb",
            family="compound_spelling",
            cases=(_case("ne_verb", "compound_spelling", "merge", "positive"),),
        ),
    )

    report = validate_layer_coverage(specs, enabled_rule_ids={"ne_verb"})

    assert report.errors == ["Layer rule 'ne_verb' must have hard_negative or clean_identity coverage."]
    assert report.warnings == []


def test_validate_layer_coverage_reports_empty_specs() -> None:
    report = validate_layer_coverage((), enabled_rule_ids={"ne_verb"})

    assert report.errors == ["No layer specs were loaded."]
    assert report.warnings == ["Enabled layer rule 'ne_verb' has no loaded spec."]


def _case(rule_id: str, family: str, sub_rule_id: str, mode: str) -> LayerDirectCase:
    return LayerDirectCase(
        rule_id=rule_id,
        family=family,
        sub_rule_id=sub_rule_id,
        mode=mode,
        source_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b.",
        target_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b.",
        expected_token_edit_count=0,
        expected_gap_edit_count=0,
    )
