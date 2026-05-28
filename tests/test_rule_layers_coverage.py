from __future__ import annotations

from pathlib import Path

import pytest

from src.config.load_config import load_config
from src.grammar_gen.factory import (
    _configured_layer_names,
    _generation_seed,
    _layer_root,
    _load_layer_specs,
    online_generator_from_config,
)
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.rule_layers.base import LayerDirectCase, LayerRuleSpec
from src.rule_layers.coverage import collect_layer_coverage, validate_layer_coverage
from src.schema.labels import RULE_LABELS


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENABLED_LAYER_GROUPS = {
    "quotation_dialogue",
    "casing",
    "semantic",
    "compound_spelling",
    "dictionary_typo",
    "syntax_punctuation",
}
GUARD_ONLY_RULE_IDS = {
    "dialogue_bracket_guards",
    "casing_formal_you_guard",
    "semantic_ne_ni",
    "semantic_introductory_context",
}


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
            metadata=_metadata(supports_positive=True, supports_hard_negative=True),
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
            metadata=_metadata(supports_positive=True),
        ),
    )

    report = validate_layer_coverage(specs, enabled_rule_ids={"ne_verb"})

    assert report.errors == ["Layer rule 'ne_verb' must have hard_negative or clean_identity coverage."]
    assert report.warnings == []


def test_validate_layer_coverage_reports_empty_specs() -> None:
    report = validate_layer_coverage((), enabled_rule_ids={"ne_verb"})

    assert report.errors == ["No layer specs were loaded."]
    assert report.warnings == ["Enabled layer rule 'ne_verb' has no loaded spec."]


def test_enabled_rule_layer_groups_load_specs_with_capabilities_metadata_and_schema_labels() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    layer_names = _configured_layer_names(config)
    root = _layer_root(config)
    seed = _generation_seed(config)
    specs = tuple(
        spec
        for layer_name in layer_names
        for spec in _load_layer_specs(layer_name, root=root, rng=RandomSource(seed=seed), seed=seed)
    )

    assert REQUIRED_ENABLED_LAYER_GROUPS <= set(layer_names)
    assert REQUIRED_ENABLED_LAYER_GROUPS <= {spec.layer for spec in specs}

    report = validate_layer_coverage(specs, enabled_rule_ids={spec.rule_id for spec in specs if spec.enabled})

    assert report.warnings == []
    assert report.errors == []

    for spec in specs:
        if spec.layer not in REQUIRED_ENABLED_LAYER_GROUPS:
            continue
        modes = {case.mode for case in spec.cases}
        assert spec.rule_id in RULE_LABELS
        assert spec.metadata
        assert spec.metadata["supports_positive"] is ("positive" in modes)
        assert spec.metadata["supports_hard_negative"] is ("hard_negative" in modes)
        assert spec.metadata["supports_clean_identity"] is ("clean_identity" in modes)
        expected_kind = "guard" if "positive" not in modes else "correction"
        assert spec.metadata["rule_kind"] == expected_kind

        if spec.rule_id in GUARD_ONLY_RULE_IDS:
            assert spec.metadata["supports_positive"] is False
            assert spec.metadata["supports_hard_negative"] is True
            assert spec.metadata["supports_clean_identity"] is True
            assert spec.metadata["rule_kind"] == "guard"
        else:
            assert spec.metadata["supports_positive"] is True
            assert {"hard_negative", "clean_identity"} & modes

        for case in spec.cases:
            assert case.metadata
            assert case.metadata["rule_id"] == spec.rule_id
            assert case.metadata["sub_rule_id"] == case.sub_rule_id


@pytest.mark.parametrize("rule_id", sorted(GUARD_ONLY_RULE_IDS))
def test_guard_only_layer_rules_generate_identity_guards_and_reject_positive(rule_id: str) -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    generator = online_generator_from_config(config, seed=13)
    rule = generator.registry.get_rule(rule_id)

    assert rule is not None
    assert not rule.can_generate(GenerationMode.POSITIVE)
    assert rule.can_generate(GenerationMode.HARD_NEGATIVE)
    assert rule.can_generate(GenerationMode.CLEAN_IDENTITY)

    for mode in (GenerationMode.HARD_NEGATIVE, GenerationMode.CLEAN_IDENTITY):
        example = generator.sample(rule_id=rule_id, mode=mode)
        assert example.source_text == example.target_text
        assert set(example.token_edit_labels) == {"KEEP"}
        assert all(
            gap_label == "NONE" or rule_token_id != rule_id
            for gap_label, rule_token_id in zip(example.gap_labels, example.rule_ids, strict=True)
        )
        assert set(example.boundary_before_labels) == {"NONE"}
        assert set(example.boundary_after_labels) == {"NONE"}
        assert example.metadata["rule_id"] == rule_id
        assert example.metadata["sub_rule_id"]
        assert validate_generated_pair(example) == []

    with pytest.raises(Exception, match="does not support mode 'positive'"):
        generator.sample(rule_id=rule_id, mode=GenerationMode.POSITIVE)


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
        metadata={"rule_id": rule_id, "sub_rule_id": sub_rule_id},
    )


def _metadata(
    *,
    supports_positive: bool,
    supports_hard_negative: bool = False,
    supports_clean_identity: bool = False,
) -> dict:
    return {
        "supports_positive": supports_positive,
        "supports_hard_negative": supports_hard_negative,
        "supports_clean_identity": supports_clean_identity,
        "rule_kind": "correction" if supports_positive else "guard",
    }
